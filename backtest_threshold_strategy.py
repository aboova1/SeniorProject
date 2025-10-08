"""
Backtest Threshold Strategy - Only Invest in High-Confidence Predictions
=========================================================================

Strategy:
1. Use train_lstm.py model (with prices) to predict expected returns
2. ONLY invest in stocks predicted to go up by MORE than 10%
3. Equal-weight investment across all qualifying stocks
4. Rebalance quarterly
5. Track returns and compare to other strategies

This tests if filtering for high-confidence predictions improves returns.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_threshold")

# ============================= Constants =====================================
BASE_EXCLUDE_COLS = [
    "sequence_id", "ticker", "quarter_in_sequence", "sequence_start_date",
    "sequence_end_date", "fiscal_quarter_end", "sector", "year",
    "transcript_date", "transcript_type", "days_after_quarter",
    "target_price_next_q", "current_price", "rebalance_date", "in_sp500"
]

# ============================= Model Architecture ============================
class CausalAttention(nn.Module):
    """Simple dot-product attention."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = hidden_size ** 0.5

    def forward(self, outputs: torch.Tensor, last_h: torch.Tensor):
        q = last_h.unsqueeze(1)
        k = outputs
        v = outputs
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / self.scale
        attn = torch.softmax(attn_logits, dim=-1)
        ctx = torch.bmm(attn, v).squeeze(1)
        return ctx, attn.squeeze(1)


class UniLSTMAttnDelta(nn.Module):
    """LSTM that predicts delta (y8_log - y7_log)."""
    def __init__(self, input_size: int, hidden_size: int = 192, num_layers: int = 2,
                 dropout: float = 0.25, use_attn: bool = True):
        super().__init__()
        self.use_attn = use_attn
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attn = CausalAttention(hidden_size) if use_attn else None
        rep_dim = hidden_size * (2 if use_attn else 1)
        self.ln = nn.LayerNorm(rep_dim)
        self.fc1 = nn.Linear(rep_dim + 1, 128)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, y7_log: torch.Tensor):
        outputs, (h, _) = self.lstm(x)
        last_h = h[-1]
        if self.use_attn:
            ctx, _ = self.attn(outputs, last_h)
            rep = torch.cat([last_h, ctx], dim=1)
        else:
            rep = last_h
        rep = self.ln(rep)
        rep = torch.cat([rep, y7_log], dim=1)
        z = self.fc1(rep)
        z = self.act(z)
        z = self.drop(z)
        delta_log_hat = self.fc2(z)
        return delta_log_hat


# ============================= Data Preparation ==============================
def load_sequences_and_prepare(
    test_path: str,
    feature_cols: List[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray
) -> pd.DataFrame:
    """Load test sequences and prepare features for prediction."""
    logger.info(f"Loading test sequences from {test_path}")
    df = pd.read_parquet(test_path)
    logger.info(f"Loaded {len(df)} rows, {df['sequence_id'].nunique()} sequences")

    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale

    # Process each sequence
    sequences = []
    for seq_id in df["sequence_id"].unique():
        seq = df[df["sequence_id"] == seq_id].sort_values("quarter_in_sequence")
        if len(seq) != 8:
            continue

        q7 = seq.iloc[6]
        q8 = seq.iloc[7]

        y7_price = q7.get('current_price', np.nan)
        y8_price = q8.get('target_price_next_q', np.nan)

        if pd.isna(y7_price) or pd.isna(y8_price):
            continue

        # Extract Q1-Q7 features
        feat_rows = seq.iloc[:7]
        X = feat_rows[feature_cols].copy()
        X = X.ffill().bfill().fillna(0.0)

        # Scale features
        X_scaled = (X.values - scaler_mean) / scaler_scale
        X_scaled = X_scaled.astype(np.float32)

        sequences.append({
            'sequence_id': seq_id,
            'ticker': q7['ticker'],
            'rebalance_date': q7.get('rebalance_date', q7.get('fiscal_quarter_end')),
            'X': X_scaled,
            'y7_price': float(y7_price),
            'y8_price_actual': float(y8_price)
        })

    logger.info(f"Prepared {len(sequences)} valid sequences for prediction")
    return pd.DataFrame(sequences)


# ============================= Prediction ====================================
def generate_predictions(model: nn.Module, sequences_df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Generate predictions for all sequences."""
    logger.info("Generating predictions...")
    model.eval()

    predictions = []
    with torch.no_grad():
        for idx, row in sequences_df.iterrows():
            X = torch.from_numpy(row['X']).unsqueeze(0).to(device)
            y7_price = row['y7_price']
            y7_log = torch.tensor([[np.log1p(y7_price)]], dtype=torch.float32).to(device)

            delta_log_pred = model(X, y7_log)
            y8_log_pred = y7_log + delta_log_pred
            y8_price_pred = float(torch.expm1(y8_log_pred).cpu().item())

            predicted_return_pct = (y8_price_pred - y7_price) / y7_price
            actual_return_pct = (row['y8_price_actual'] - y7_price) / y7_price

            predictions.append({
                'sequence_id': row['sequence_id'],
                'ticker': row['ticker'],
                'rebalance_date': row['rebalance_date'],
                'y7_price': y7_price,
                'y8_price_actual': row['y8_price_actual'],
                'y8_price_predicted': y8_price_pred,
                'predicted_return_pct': predicted_return_pct,
                'actual_return_pct': actual_return_pct
            })

    logger.info(f"Generated {len(predictions)} predictions")
    return pd.DataFrame(predictions)


# ============================= Backtesting ===================================
def backtest_threshold_strategy(
    predictions_df: pd.DataFrame,
    threshold_pct: float = 0.10,
    total_capital_per_quarter: float = 10000.0
) -> Dict:
    """
    Backtest threshold strategy: only invest in stocks predicted to gain > threshold.

    Strategy:
    - Filter to stocks with predicted return > threshold (e.g., 10%)
    - Equal-weight investment across qualifying stocks
    - Rebalance quarterly

    Args:
        predictions_df: DataFrame with predictions
        threshold_pct: Minimum predicted return to invest (default: 0.10 = 10%)
        total_capital_per_quarter: Total capital to invest each quarter

    Returns:
        Dictionary with backtest results
    """
    logger.info("="*70)
    logger.info(f"BACKTESTING THRESHOLD STRATEGY (>{threshold_pct*100:.0f}% PREDICTED GAIN)")
    logger.info("="*70)
    logger.info(f"Total capital per quarter: ${total_capital_per_quarter:,.2f}")

    # Sort by rebalance date
    predictions_df['rebalance_date'] = pd.to_datetime(predictions_df['rebalance_date'])
    predictions_df = predictions_df.sort_values('rebalance_date')

    quarterly_results = []
    total_invested_all = 0.0
    total_returns_all = 0.0

    for quarter_date, quarter_group in predictions_df.groupby('rebalance_date'):
        # Filter to stocks predicted to gain > threshold
        high_confidence = quarter_group[quarter_group['predicted_return_pct'] > threshold_pct].copy()

        if len(high_confidence) == 0:
            logger.warning(f"Quarter {quarter_date}: No stocks above {threshold_pct*100:.0f}% threshold, skipping")
            continue

        # Equal-weight investment
        num_stocks = len(high_confidence)
        investment_per_stock = total_capital_per_quarter / num_stocks
        high_confidence['investment'] = investment_per_stock

        # Calculate actual returns
        high_confidence['dollar_return'] = high_confidence['investment'] * high_confidence['actual_return_pct']

        quarter_invested = high_confidence['investment'].sum()
        quarter_return = high_confidence['dollar_return'].sum()
        quarter_roi = (quarter_return / quarter_invested * 100) if quarter_invested > 0 else 0

        total_invested_all += quarter_invested
        total_returns_all += quarter_return

        # Count correct predictions (actually went up)
        correct_predictions = (high_confidence['actual_return_pct'] > 0).sum()
        total_predictions = len(high_confidence)
        accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0

        # Calculate average predicted vs actual return
        avg_predicted = high_confidence['predicted_return_pct'].mean() * 100
        avg_actual = high_confidence['actual_return_pct'].mean() * 100

        quarterly_results.append({
            'quarter_date': str(quarter_date),
            'num_positions': total_predictions,
            'capital_invested': float(quarter_invested),
            'total_return': float(quarter_return),
            'roi_pct': float(quarter_roi),
            'accuracy_pct': float(accuracy),
            'correct_predictions': int(correct_predictions),
            'avg_predicted_return_pct': float(avg_predicted),
            'avg_actual_return_pct': float(avg_actual)
        })

    # Calculate overall statistics
    total_quarters = len(quarterly_results)
    avg_return_per_quarter = total_returns_all / total_quarters if total_quarters > 0 else 0
    overall_roi = (total_returns_all / total_invested_all * 100) if total_invested_all > 0 else 0

    total_accuracy = sum(q['correct_predictions'] for q in quarterly_results)
    total_predictions = sum(q['num_positions'] for q in quarterly_results)
    overall_accuracy = (total_accuracy / total_predictions * 100) if total_predictions > 0 else 0

    # Calculate average position size
    avg_positions_per_quarter = total_predictions / total_quarters if total_quarters > 0 else 0

    # Calculate yearly ROI first (needed for annualized calculation)
    logger.info("\nCalculating yearly ROI...")
    yearly_data = {}
    for q in quarterly_results:
        quarter_date = pd.to_datetime(q['quarter_date'])
        year = quarter_date.year
        if year not in yearly_data:
            yearly_data[year] = {'invested': 0.0, 'returns': 0.0, 'positions': 0}

        yearly_data[year]['invested'] += q['capital_invested']
        yearly_data[year]['returns'] += q['total_return']
        yearly_data[year]['positions'] += q['num_positions']

    yearly_roi = []
    for year in sorted(yearly_data.keys()):
        invested = yearly_data[year]['invested']
        returns = yearly_data[year]['returns']
        roi_pct = (returns / invested * 100) if invested > 0 else 0
        yearly_roi.append({
            'year': year,
            'invested': float(invested),
            'returns': float(returns),
            'roi_pct': float(roi_pct),
            'positions': yearly_data[year]['positions']
        })

    # Calculate annualized ROI as average of yearly ROIs
    num_years = len(yearly_roi)
    annualized_roi = sum(yr['roi_pct'] for yr in yearly_roi) / num_years if num_years > 0 else 0

    # Calculate sample standard deviation of yearly ROIs (using N-1 for Bessel's correction)
    if num_years > 1:
        yearly_roi_values = [yr['roi_pct'] for yr in yearly_roi]
        variance = sum((x - annualized_roi) ** 2 for x in yearly_roi_values) / (num_years - 1)
        std_dev = variance ** 0.5
    else:
        std_dev = 0.0

    # Log summary
    logger.info(f"\nBACKTEST PERIOD: {quarterly_results[0]['quarter_date']} to {quarterly_results[-1]['quarter_date']}")
    logger.info(f"Total Quarters: {total_quarters}")
    logger.info(f"Total Positions Taken: {total_predictions:,}")
    logger.info(f"Avg Positions per Quarter: {avg_positions_per_quarter:.1f}")
    logger.info(f"Total Capital Invested: ${total_invested_all:,.2f}")
    logger.info(f"Total Returns: ${total_returns_all:,.2f}")
    logger.info(f"Average Return per Quarter: ${avg_return_per_quarter:,.2f}")
    logger.info(f"Overall ROI: {overall_roi:.2f}%")
    logger.info(f"Annualized ROI (avg of yearly): {annualized_roi:.2f}%")
    logger.info(f"Annualized Std Dev: {std_dev:.2f}%")
    logger.info(f"Directional Accuracy: {overall_accuracy:.2f}% ({total_accuracy}/{total_predictions})")
    logger.info("="*70)

    logger.info("\nYearly ROI:")
    for yr in yearly_roi:
        logger.info(f"  {yr['year']}: {yr['roi_pct']:+.2f}% ({yr['positions']} positions, ${yr['returns']:,.2f} / ${yr['invested']:,.2f})")
    logger.info("="*70)

    return {
        'summary': {
            'threshold_pct': float(threshold_pct * 100),
            'total_quarters': total_quarters,
            'total_positions': total_predictions,
            'avg_positions_per_quarter': float(avg_positions_per_quarter),
            'total_invested': float(total_invested_all),
            'total_returns': float(total_returns_all),
            'avg_return_per_quarter': float(avg_return_per_quarter),
            'overall_roi_pct': float(overall_roi),
            'annualized_roi_pct': float(annualized_roi),
            'annualized_std_dev_pct': float(std_dev),
            'num_years': float(num_years),
            'directional_accuracy_pct': float(overall_accuracy),
            'correct_predictions': int(total_accuracy)
        },
        'quarterly_results': quarterly_results,
        'yearly_roi': yearly_roi
    }


# ============================= Main ==========================================
def main():
    parser = argparse.ArgumentParser(description="Backtest threshold strategy (only invest in high-confidence predictions)")
    parser.add_argument("--test", type=str, default="data_pipeline/data/sequences_8q_test.parquet",
                        help="Path to test sequences parquet")
    parser.add_argument("--model", type=str, default="models/best_model.pt",
                        help="Path to trained model checkpoint (train_lstm.py)")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Minimum predicted return threshold (default: 0.10 = 10%%)")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Total capital to invest per quarter (default: $10,000)")
    parser.add_argument("--output", type=str, default="backtest_threshold_results.json",
                        help="Output JSON file for backtest results")
    args = parser.parse_args()

    # Check files exist
    test_path = Path(args.test)
    model_path = Path(args.model)
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    feature_cols = checkpoint['feature_cols']
    scaler_mean = checkpoint['scaler_mean']
    scaler_scale = checkpoint['scaler_scale']
    use_attn = checkpoint.get('use_attn', True)

    logger.info(f"Model uses {len(feature_cols)} features")
    logger.info(f"Attention: {use_attn}")

    # Create model and load weights
    model = UniLSTMAttnDelta(
        input_size=len(feature_cols),
        hidden_size=192,
        num_layers=2,
        dropout=0.25,
        use_attn=use_attn
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully")

    # Load and prepare data
    sequences_df = load_sequences_and_prepare(
        str(test_path),
        feature_cols,
        scaler_mean,
        scaler_scale
    )

    # Generate predictions
    predictions_df = generate_predictions(model, sequences_df, device)

    # Run backtest
    backtest_results = backtest_threshold_strategy(
        predictions_df,
        threshold_pct=args.threshold,
        total_capital_per_quarter=args.capital
    )

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(backtest_results, f, indent=2)
    logger.info(f"\nBacktest results saved to {output_path}")

    # Print final summary
    summary = backtest_results['summary']
    print("\n" + "="*70)
    print(f"FINAL BACKTEST SUMMARY - THRESHOLD STRATEGY (>{summary['threshold_pct']:.0f}%)")
    print("="*70)
    print(f"Backtest Period: {summary['num_years']:.1f} years ({summary['total_quarters']} quarters)")
    print(f"Total Positions Taken: {summary['total_positions']:,}")
    print(f"Avg Positions per Quarter: {summary['avg_positions_per_quarter']:.1f}")
    print(f"Total Capital Invested: ${summary['total_invested']:,.2f}")
    print(f"Total Returns: ${summary['total_returns']:,.2f}")
    print(f"Average Return per Quarter: ${summary['avg_return_per_quarter']:,.2f}")
    print(f"\nROI Metrics:")
    print(f"  Overall ROI (total): {summary['overall_roi_pct']:.2f}%")
    print(f"  Annualized ROI: {summary['annualized_roi_pct']:.2f}%")
    print(f"  Annualized Std Dev: {summary['annualized_std_dev_pct']:.2f}%")
    print(f"\nDirectional Accuracy: {summary['directional_accuracy_pct']:.2f}%")
    print(f"  ({summary['correct_predictions']}/{summary['total_positions']} predictions correct)")

    # Print yearly ROI
    print(f"\nYearly ROI:")
    for year_data in backtest_results['yearly_roi']:
        print(f"  {year_data['year']}: {year_data['roi_pct']:+.2f}%")

    print("="*70)


if __name__ == "__main__":
    main()

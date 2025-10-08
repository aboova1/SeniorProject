"""
Backtest Proportional Investment Strategy with Dual Model Consensus
====================================================================

Strategy:
1. Use BOTH models to predict stock movement:
   - train_lstm.py (with prices) - predicts expected price change
   - train_lstm_no_price.py - predicts direction
2. ONLY invest if BOTH models predict stock will go UP
3. Investment amount is proportional to predicted return magnitude from price model
4. Rebalance quarterly
5. Track total returns over 20-year test period

Investment allocation:
- Both models must predict UP, otherwise $0 investment
- If both agree on UP and price model predicts +10%, invest proportionally more
- If both agree on UP and price model predicts +2%, invest proportionally less
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
logger = logging.getLogger("backtest_proportional")

# ============================= Constants =====================================
BASE_EXCLUDE_COLS = [
    "sequence_id", "ticker", "quarter_in_sequence", "sequence_start_date",
    "sequence_end_date", "fiscal_quarter_end", "sector", "year",
    "transcript_date", "transcript_type", "days_after_quarter",
    "target_price_next_q", "current_price", "rebalance_date", "in_sp500"
]

PRICE_CANDIDATE_NAMES = [
    "quarter_end_adj_close", "quarter_end_price", "adj_close", "close", "price", "px_last",
    "last_price", "prc", "mkt_cap", "marketcapitalization"
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
    """LSTM that predicts delta (y8_log - y7_log) - WITH PRICE model."""
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
        self.fc1 = nn.Linear(rep_dim + 1, 128)  # +1 for y7_log
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


class UniLSTMAttnDirect(nn.Module):
    """LSTM that predicts Q8 price DIRECTLY (no Q7 price input) - NO PRICE model."""
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
        self.fc1 = nn.Linear(rep_dim, 128)  # NO +1 for y7_log!
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        outputs, (h, _) = self.lstm(x)
        last_h = h[-1]
        if self.use_attn:
            ctx, _ = self.attn(outputs, last_h)
            rep = torch.cat([last_h, ctx], dim=1)
        else:
            rep = last_h
        rep = self.ln(rep)
        z = self.fc1(rep)
        z = self.act(z)
        z = self.drop(z)
        y8_log_hat = self.fc2(z)
        return y8_log_hat


# ============================= Data Preparation ==============================
def load_sequences_and_prepare_dual(
    test_path: str,
    price_feature_cols: List[str],
    price_scaler_mean: np.ndarray,
    price_scaler_scale: np.ndarray,
    no_price_feature_cols: List[str],
    no_price_scaler_mean: np.ndarray,
    no_price_scaler_scale: np.ndarray
) -> pd.DataFrame:
    """
    Load test sequences and prepare features for BOTH models.
    """
    logger.info(f"Loading test sequences from {test_path}")
    df = pd.read_parquet(test_path)
    logger.info(f"Loaded {len(df)} rows, {df['sequence_id'].nunique()} sequences")

    # Reconstruct scalers
    price_scaler = StandardScaler()
    price_scaler.mean_ = price_scaler_mean
    price_scaler.scale_ = price_scaler_scale

    no_price_scaler = StandardScaler()
    no_price_scaler.mean_ = no_price_scaler_mean
    no_price_scaler.scale_ = no_price_scaler_scale

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

        # Prepare price model features
        X_price = feat_rows[price_feature_cols].copy()
        X_price = X_price.ffill().bfill().fillna(0.0)
        X_price_scaled = (X_price.values - price_scaler_mean) / price_scaler_scale
        X_price_scaled = X_price_scaled.astype(np.float32)

        # Prepare no-price model features
        X_no_price = feat_rows[no_price_feature_cols].copy()
        X_no_price = X_no_price.ffill().bfill().fillna(0.0)
        X_no_price_scaled = (X_no_price.values - no_price_scaler_mean) / no_price_scaler_scale
        X_no_price_scaled = X_no_price_scaled.astype(np.float32)

        sequences.append({
            'sequence_id': seq_id,
            'ticker': q7['ticker'],
            'rebalance_date': q7.get('rebalance_date', q7.get('fiscal_quarter_end')),
            'X': X_price_scaled,  # For price model
            'X_no_price': X_no_price_scaled,  # For no-price model
            'y7_price': float(y7_price),
            'y8_price_actual': float(y8_price)
        })

    logger.info(f"Prepared {len(sequences)} valid sequences for prediction")
    return pd.DataFrame(sequences)


# ============================= Prediction ====================================
def generate_predictions_dual_model(
    price_model: nn.Module,
    no_price_model: nn.Module,
    sequences_df: pd.DataFrame,
    price_feature_cols: List[str],
    no_price_feature_cols: List[str],
    no_price_scaler_mean: np.ndarray,
    no_price_scaler_scale: np.ndarray,
    device: torch.device
) -> pd.DataFrame:
    """
    Generate predictions using BOTH models and only keep stocks where both predict UP.

    Returns:
        DataFrame with predictions from both models, filtered to consensus UP predictions.
    """
    logger.info("Generating predictions from BOTH models...")
    price_model.eval()
    no_price_model.eval()

    predictions = []
    with torch.no_grad():
        for idx, row in sequences_df.iterrows():
            # ===== PRICE MODEL PREDICTION =====
            X_price = torch.from_numpy(row['X']).unsqueeze(0).to(device)  # (1, 7, F)
            y7_price = row['y7_price']
            y7_log = torch.tensor([[np.log1p(y7_price)]], dtype=torch.float32).to(device)

            delta_log_pred = price_model(X_price, y7_log)  # (1, 1)
            y8_log_pred = y7_log + delta_log_pred
            y8_price_pred = float(torch.expm1(y8_log_pred).cpu().item())
            price_model_predicted_return = (y8_price_pred - y7_price) / y7_price

            # ===== NO-PRICE MODEL PREDICTION =====
            # Need to prepare features for no-price model (different feature set)
            X_no_price = torch.from_numpy(row['X_no_price']).unsqueeze(0).to(device)  # (1, 7, F_no_price)
            y8_log_pred_no_price = no_price_model(X_no_price)  # (1, 1)
            y8_price_pred_no_price = float(torch.expm1(y8_log_pred_no_price).cpu().item())
            no_price_model_predicted_return = (y8_price_pred_no_price - y7_price) / y7_price

            # ===== ACTUAL RETURN =====
            actual_return_pct = (row['y8_price_actual'] - y7_price) / y7_price

            # ===== CHECK CONSENSUS =====
            price_model_predicts_up = price_model_predicted_return > 0
            no_price_model_predicts_up = no_price_model_predicted_return > 0
            both_predict_up = price_model_predicts_up and no_price_model_predicts_up

            predictions.append({
                'sequence_id': row['sequence_id'],
                'ticker': row['ticker'],
                'rebalance_date': row['rebalance_date'],
                'y7_price': y7_price,
                'y8_price_actual': row['y8_price_actual'],
                'price_model_predicted_return': price_model_predicted_return,
                'no_price_model_predicted_return': no_price_model_predicted_return,
                'both_predict_up': both_predict_up,
                'actual_return_pct': actual_return_pct
            })

    predictions_df = pd.DataFrame(predictions)
    logger.info(f"Generated {len(predictions_df)} total predictions")
    logger.info(f"  - Price model predicts UP: {(predictions_df['price_model_predicted_return'] > 0).sum()}")
    logger.info(f"  - No-price model predicts UP: {(predictions_df['no_price_model_predicted_return'] > 0).sum()}")
    logger.info(f"  - BOTH models predict UP (consensus): {predictions_df['both_predict_up'].sum()}")

    return predictions_df


# ============================= Backtesting ===================================
def backtest_proportional_strategy(
    predictions_df: pd.DataFrame,
    total_capital_per_quarter: float = 10000.0
) -> Dict:
    """
    Backtest proportional investment strategy with DUAL MODEL CONSENSUS.

    Strategy:
    - Each quarter, ONLY invest in stocks where BOTH models predict UP
    - Investment amount proportional to predicted return from price model
    - Stocks where models disagree or both predict down get $0 investment

    Args:
        predictions_df: DataFrame with predictions from both models
        total_capital_per_quarter: Total capital to invest each quarter (default $10,000)

    Returns:
        Dictionary with backtest results
    """
    logger.info("="*70)
    logger.info("BACKTESTING PROPORTIONAL INVESTMENT WITH DUAL MODEL CONSENSUS")
    logger.info("="*70)
    logger.info(f"Total capital per quarter: ${total_capital_per_quarter:,.2f}")

    # Sort by rebalance date
    predictions_df['rebalance_date'] = pd.to_datetime(predictions_df['rebalance_date'])
    predictions_df = predictions_df.sort_values('rebalance_date')

    quarterly_results = []
    total_invested_all = 0.0
    total_returns_all = 0.0

    for quarter_date, quarter_group in predictions_df.groupby('rebalance_date'):
        # Filter to ONLY stocks where BOTH models predict UP
        consensus_up = quarter_group[quarter_group['both_predict_up'] == True].copy()

        if len(consensus_up) == 0:
            logger.warning(f"Quarter {quarter_date}: No stocks with consensus UP prediction, skipping")
            continue

        # Calculate investment weights based on price model's predicted return magnitude
        consensus_up['weight'] = consensus_up['price_model_predicted_return']
        total_weight = consensus_up['weight'].sum()

        # Allocate capital proportionally
        consensus_up['investment'] = (consensus_up['weight'] / total_weight) * total_capital_per_quarter

        # Calculate actual returns
        consensus_up['dollar_return'] = consensus_up['investment'] * consensus_up['actual_return_pct']

        quarter_invested = consensus_up['investment'].sum()
        quarter_return = consensus_up['dollar_return'].sum()
        quarter_roi = (quarter_return / quarter_invested * 100) if quarter_invested > 0 else 0

        total_invested_all += quarter_invested
        total_returns_all += quarter_return

        # Count correct predictions
        correct_predictions = (consensus_up['actual_return_pct'] > 0).sum()
        total_predictions = len(consensus_up)
        accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0

        quarterly_results.append({
            'quarter_date': str(quarter_date),
            'num_positions': total_predictions,
            'capital_invested': float(quarter_invested),
            'total_return': float(quarter_return),
            'roi_pct': float(quarter_roi),
            'accuracy_pct': float(accuracy),
            'correct_predictions': int(correct_predictions)
        })

    # Calculate overall statistics
    total_quarters = len(quarterly_results)
    avg_return_per_quarter = total_returns_all / total_quarters if total_quarters > 0 else 0
    overall_roi = (total_returns_all / total_invested_all * 100) if total_invested_all > 0 else 0

    total_accuracy = sum(q['correct_predictions'] for q in quarterly_results)
    total_predictions = sum(q['num_positions'] for q in quarterly_results)
    overall_accuracy = (total_accuracy / total_predictions * 100) if total_predictions > 0 else 0

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
            'total_quarters': total_quarters,
            'total_positions': total_predictions,
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
    parser = argparse.ArgumentParser(description="Backtest proportional investment with dual model consensus")
    parser.add_argument("--test", type=str, default="data_pipeline/data/sequences_8q_test.parquet",
                        help="Path to test sequences parquet")
    parser.add_argument("--price-model", type=str, default="models/best_model.pt",
                        help="Path to price model checkpoint (train_lstm.py)")
    parser.add_argument("--no-price-model", type=str, default="models_no_price/best_model_no_price.pt",
                        help="Path to no-price model checkpoint (train_lstm_no_price.py)")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Total capital to invest per quarter (default: $10,000)")
    parser.add_argument("--output", type=str, default="backtest_dual_consensus_results.json",
                        help="Output JSON file for backtest results")
    args = parser.parse_args()

    # Check files exist
    test_path = Path(args.test)
    price_model_path = Path(args.price_model)
    no_price_model_path = Path(args.no_price_model)

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    if not price_model_path.exists():
        raise FileNotFoundError(f"Price model not found: {price_model_path}")
    if not no_price_model_path.exists():
        raise FileNotFoundError(f"No-price model not found: {no_price_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load PRICE model
    logger.info(f"Loading PRICE model from {price_model_path}")
    price_ckpt = torch.load(price_model_path, map_location=device, weights_only=False)
    price_feature_cols = price_ckpt['feature_cols']
    price_scaler_mean = price_ckpt['scaler_mean']
    price_scaler_scale = price_ckpt['scaler_scale']
    price_use_attn = price_ckpt.get('use_attn', True)
    logger.info(f"  Price model: {len(price_feature_cols)} features, attention={price_use_attn}")

    price_model = UniLSTMAttnDelta(
        input_size=len(price_feature_cols),
        hidden_size=192,
        num_layers=2,
        dropout=0.25,
        use_attn=price_use_attn
    ).to(device)
    price_model.load_state_dict(price_ckpt['model_state_dict'])
    price_model.eval()
    logger.info("  Price model loaded successfully")

    # Load NO-PRICE model
    logger.info(f"Loading NO-PRICE model from {no_price_model_path}")
    no_price_ckpt = torch.load(no_price_model_path, map_location=device, weights_only=False)
    no_price_feature_cols = no_price_ckpt['feature_cols']
    no_price_scaler_mean = no_price_ckpt['scaler_mean']
    no_price_scaler_scale = no_price_ckpt['scaler_scale']
    no_price_use_attn = no_price_ckpt.get('use_attn', True)
    logger.info(f"  No-price model: {len(no_price_feature_cols)} features, attention={no_price_use_attn}")

    no_price_model = UniLSTMAttnDirect(
        input_size=len(no_price_feature_cols),
        hidden_size=192,
        num_layers=2,
        dropout=0.25,
        use_attn=no_price_use_attn
    ).to(device)
    no_price_model.load_state_dict(no_price_ckpt['model_state_dict'])
    no_price_model.eval()
    logger.info("  No-price model loaded successfully")

    # Load and prepare data for BOTH models
    sequences_df = load_sequences_and_prepare_dual(
        str(test_path),
        price_feature_cols,
        price_scaler_mean,
        price_scaler_scale,
        no_price_feature_cols,
        no_price_scaler_mean,
        no_price_scaler_scale
    )

    # Generate predictions from BOTH models
    predictions_df = generate_predictions_dual_model(
        price_model,
        no_price_model,
        sequences_df,
        price_feature_cols,
        no_price_feature_cols,
        no_price_scaler_mean,
        no_price_scaler_scale,
        device
    )

    # Run backtest
    backtest_results = backtest_proportional_strategy(
        predictions_df,
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
    print("FINAL BACKTEST SUMMARY - DUAL MODEL CONSENSUS STRATEGY")
    print("="*70)
    print(f"Backtest Period: {summary['num_years']:.1f} years ({summary['total_quarters']} quarters)")
    print(f"Total Positions Taken: {summary['total_positions']:,}")
    print(f"Total Capital Invested: ${summary['total_invested']:,.2f}")
    print(f"Total Returns: ${summary['total_returns']:,.2f}")
    print(f"Average Return per Quarter: ${summary['avg_return_per_quarter']:,.2f}")
    print(f"\nROI Metrics:")
    print(f"  Overall ROI (total): {summary['overall_roi_pct']:.2f}%")
    print(f"  Annualized ROI: {summary['annualized_roi_pct']:.2f}%")
    print(f"  Annualized Std Dev: {summary['annualized_std_dev_pct']:.2f}%")
    print(f"\nDirectional Accuracy: {summary['directional_accuracy_pct']:.2f}%")
    print(f"  ({summary['correct_predictions']}/{summary['total_positions']} predictions correct)")

    # Print yearly ROI if available
    if 'yearly_roi' in backtest_results:
        print(f"\nYearly ROI:")
        for year_data in backtest_results['yearly_roi']:
            print(f"  {year_data['year']}: {year_data['roi_pct']:+.2f}%")

    print("="*70)


if __name__ == "__main__":
    main()

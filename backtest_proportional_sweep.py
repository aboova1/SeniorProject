"""
Backtest Proportional Threshold Sweep - Test Multiple Thresholds
=================================================================

Strategy:
Run the proportional threshold strategy for multiple threshold values (0.01 to 0.30 in 0.01 increments)
to find the optimal threshold.

Investment is proportional to predicted return magnitude (not fixed per stock).
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
logger = logging.getLogger("backtest_proportional_sweep")

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
def backtest_single_threshold(
    predictions_df: pd.DataFrame,
    threshold_pct: float,
    total_capital_per_quarter: float = 10000.0
) -> Dict:
    """Backtest a single threshold value with PROPORTIONAL investment (silent mode - no logging).

    Args:
        predictions_df: DataFrame with predictions
        threshold_pct: Minimum predicted return threshold
        total_capital_per_quarter: Total capital to allocate per quarter (default: $10,000)
    """

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
            continue

        # PROPORTIONAL investment based on predicted return magnitude
        high_confidence['weight'] = high_confidence['predicted_return_pct']
        total_weight = high_confidence['weight'].sum()

        # Allocate capital proportionally
        high_confidence['investment'] = (high_confidence['weight'] / total_weight) * total_capital_per_quarter

        # Calculate actual returns
        high_confidence['dollar_return'] = high_confidence['investment'] * high_confidence['actual_return_pct']

        quarter_invested = high_confidence['investment'].sum()
        quarter_return = high_confidence['dollar_return'].sum()

        total_invested_all += quarter_invested
        total_returns_all += quarter_return

        # Count correct predictions
        correct_predictions = (high_confidence['actual_return_pct'] > 0).sum()

        quarterly_results.append({
            'quarter_date': str(quarter_date),
            'num_positions': len(high_confidence),
            'capital_invested': float(quarter_invested),
            'total_return': float(quarter_return),
            'correct_predictions': int(correct_predictions)
        })

    if len(quarterly_results) == 0:
        return None

    # Calculate overall statistics
    total_quarters = len(quarterly_results)
    overall_roi = (total_returns_all / total_invested_all * 100) if total_invested_all > 0 else 0

    total_accuracy = sum(q['correct_predictions'] for q in quarterly_results)
    total_predictions = sum(q['num_positions'] for q in quarterly_results)
    overall_accuracy = (total_accuracy / total_predictions * 100) if total_predictions > 0 else 0

    # Calculate yearly ROI
    yearly_data = {}
    for q in quarterly_results:
        quarter_date = pd.to_datetime(q['quarter_date'])
        year = quarter_date.year
        if year not in yearly_data:
            yearly_data[year] = {'invested': 0.0, 'returns': 0.0}

        yearly_data[year]['invested'] += q['capital_invested']
        yearly_data[year]['returns'] += q['total_return']

    yearly_roi = []
    for year in sorted(yearly_data.keys()):
        invested = yearly_data[year]['invested']
        returns = yearly_data[year]['returns']
        roi_pct = (returns / invested * 100) if invested > 0 else 0
        yearly_roi.append(roi_pct)

    # Calculate annualized ROI and std dev
    num_years = len(yearly_roi)
    annualized_roi = sum(yearly_roi) / num_years if num_years > 0 else 0

    if num_years > 1:
        variance = sum((x - annualized_roi) ** 2 for x in yearly_roi) / (num_years - 1)
        std_dev = variance ** 0.5
    else:
        std_dev = 0.0

    avg_positions_per_quarter = total_predictions / total_quarters if total_quarters > 0 else 0

    return {
        'threshold_pct': float(threshold_pct * 100),
        'total_quarters': total_quarters,
        'total_positions': total_predictions,
        'avg_positions_per_quarter': float(avg_positions_per_quarter),
        'total_invested': float(total_invested_all),
        'total_returns': float(total_returns_all),
        'overall_roi_pct': float(overall_roi),
        'annualized_roi_pct': float(annualized_roi),
        'annualized_std_dev_pct': float(std_dev),
        'directional_accuracy_pct': float(overall_accuracy),
        'num_years': num_years
    }


# ============================= Main ==========================================
def main():
    parser = argparse.ArgumentParser(description="Sweep threshold values with PROPORTIONAL investment")
    parser.add_argument("--test", type=str, default="data_pipeline/data/sequences_8q_test.parquet",
                        help="Path to test sequences parquet")
    parser.add_argument("--model", type=str, default="models/best_model.pt",
                        help="Path to trained model checkpoint (train_lstm.py)")
    parser.add_argument("--min-threshold", type=float, default=0.01,
                        help="Minimum threshold to test (default: 0.01 = 1%%)")
    parser.add_argument("--max-threshold", type=float, default=0.30,
                        help="Maximum threshold to test (default: 0.30 = 30%%)")
    parser.add_argument("--step", type=float, default=0.01,
                        help="Step size for threshold sweep (default: 0.01 = 1%%)")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Total capital to invest per quarter (default: $10,000)")
    parser.add_argument("--output", type=str, default="backtest_proportional_sweep_results.json",
                        help="Output JSON file for sweep results")
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

    # Load and prepare data (only once!)
    sequences_df = load_sequences_and_prepare(
        str(test_path),
        feature_cols,
        scaler_mean,
        scaler_scale
    )

    # Generate predictions (only once!)
    predictions_df = generate_predictions(model, sequences_df, device)

    # Sweep thresholds
    thresholds = np.arange(args.min_threshold, args.max_threshold + args.step/2, args.step)
    logger.info("="*70)
    logger.info(f"SWEEPING THRESHOLDS FROM {args.min_threshold*100:.0f}% TO {args.max_threshold*100:.0f}% (STEP: {args.step*100:.0f}%)")
    logger.info(f"Strategy: PROPORTIONAL INVESTMENT (capital allocated by predicted return)")
    logger.info(f"Total capital per quarter: ${args.capital:,.0f}")
    logger.info("="*70)

    results = []
    for threshold in thresholds:
        logger.info(f"Testing threshold: {threshold*100:.0f}%...")
        result = backtest_single_threshold(predictions_df, threshold, args.capital)
        if result is not None:
            results.append(result)
            logger.info(f"  ✓ Annualized ROI: {result['annualized_roi_pct']:.2f}%, Std Dev: {result['annualized_std_dev_pct']:.2f}%, Positions/Q: {result['avg_positions_per_quarter']:.1f}")
        else:
            logger.info(f"  ✗ No qualifying trades for this threshold")

    # Find best threshold
    if results:
        best_result = max(results, key=lambda x: x['annualized_roi_pct'])

        logger.info("="*70)
        logger.info("THRESHOLD SWEEP COMPLETE")
        logger.info("="*70)
        logger.info(f"Tested {len(results)} threshold values")
        logger.info(f"\nBEST THRESHOLD: {best_result['threshold_pct']:.0f}%")
        logger.info(f"  Annualized ROI: {best_result['annualized_roi_pct']:.2f}%")
        logger.info(f"  Annualized Std Dev: {best_result['annualized_std_dev_pct']:.2f}%")
        logger.info(f"  Avg Positions/Quarter: {best_result['avg_positions_per_quarter']:.1f}")
        logger.info(f"  Directional Accuracy: {best_result['directional_accuracy_pct']:.2f}%")
        logger.info("="*70)

        # Save results
        output_data = {
            'sweep_parameters': {
                'min_threshold': args.min_threshold * 100,
                'max_threshold': args.max_threshold * 100,
                'step': args.step * 100,
                'capital_per_quarter': args.capital,
                'strategy': 'proportional'
            },
            'best_threshold': best_result,
            'all_results': results
        }

        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")

        # Print summary table
        print("\n" + "="*100)
        print("PROPORTIONAL THRESHOLD SWEEP SUMMARY")
        print("="*100)
        print(f"{'Threshold':>10} {'Ann. ROI':>10} {'Std Dev':>10} {'Positions/Q':>12} {'Accuracy':>10} {'Total Pos':>10}")
        print("-"*100)
        for r in sorted(results, key=lambda x: x['threshold_pct']):
            print(f"{r['threshold_pct']:>9.0f}% {r['annualized_roi_pct']:>9.2f}% {r['annualized_std_dev_pct']:>9.2f}% {r['avg_positions_per_quarter']:>11.1f} {r['directional_accuracy_pct']:>9.1f}% {r['total_positions']:>10}")
        print("="*100)
        print(f"BEST: {best_result['threshold_pct']:.0f}% threshold with {best_result['annualized_roi_pct']:.2f}% annualized ROI")
        print("="*100)
    else:
        logger.error("No valid results found for any threshold!")


if __name__ == "__main__":
    main()

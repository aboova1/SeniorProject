"""
Backtest Q2 2025 - Recent Quarter Analysis
==========================================

Shows which stocks the model would have invested in during Q2 2025
to sell at Q3 2025 prices, testing thresholds from 5% to 25%.

This provides real-world validation of the model's recent performance.

IMPORTANT: Uses Company-Specific Rebalance Timing
--------------------------------------------------
- Each company has a unique rebalance lag based on their historical transcript timing
- Rebalance dates range from 18-87 days after fiscal quarter end (avg: 30.2 days)
- Per-company lag = p75 of historical transcript delays + 7-day buffer
- This avoids look-ahead bias by only using information when it would be available
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_q2_2025")

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
def load_q2_2025_sequences(
    test_path: str,
    feature_cols: List[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray
) -> pd.DataFrame:
    """Load sequences with Q2 2025 as the rebalance quarter from ALL data splits."""
    # Load all three splits to get all S&P 500 stocks
    data_dir = Path(test_path).parent
    train_path = data_dir / "sequences_8q_train.parquet"
    val_path = data_dir / "sequences_8q_val.parquet"
    test_path_obj = Path(test_path)

    logger.info(f"Loading sequences from train, val, and test sets...")
    dfs = []

    if train_path.exists():
        train_df = pd.read_parquet(train_path)
        dfs.append(train_df)
        logger.info(f"  Train: {len(train_df)} rows, {train_df['ticker'].nunique()} tickers")

    if val_path.exists():
        val_df = pd.read_parquet(val_path)
        dfs.append(val_df)
        logger.info(f"  Val: {len(val_df)} rows, {val_df['ticker'].nunique()} tickers")

    if test_path_obj.exists():
        test_df = pd.read_parquet(test_path_obj)
        dfs.append(test_df)
        logger.info(f"  Test: {len(test_df)} rows, {test_df['ticker'].nunique()} tickers")

    # Combine all datasets
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined: {len(df)} rows, {df['sequence_id'].nunique()} sequences, {df['ticker'].nunique()} unique tickers")

    # Calculate period start timing statistics (when transcript becomes available)
    df['fiscal_quarter_end'] = pd.to_datetime(df['fiscal_quarter_end'])
    df['period_start_date'] = pd.to_datetime(df.get('period_start_date', df.get('rebalance_date')))
    df['days_to_period_start'] = (df['period_start_date'] - df['fiscal_quarter_end']).dt.days

    timing_stats = df.groupby('ticker')['days_to_period_start'].first()
    logger.info(f"Period start timing: Min={timing_stats.min():.0f}d, Max={timing_stats.max():.0f}d, Mean={timing_stats.mean():.1f}d, Median={timing_stats.median():.0f}d")

    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale

    # Filter for Q2 2025 (April-June 2025)
    sequences = []
    for seq_id in df["sequence_id"].unique():
        seq = df[df["sequence_id"] == seq_id].sort_values("quarter_in_sequence")
        if len(seq) != 8:
            continue

        q7 = seq.iloc[6]
        q8 = seq.iloc[7]

        # Parse period start date (when transcript becomes available)
        period_start_date = pd.to_datetime(q7.get('period_start_date', q7.get('rebalance_date', q7.get('fiscal_quarter_end'))))

        # Filter for Q2 2025 (April 1 - June 30, 2025)
        if not (period_start_date.year == 2025 and 4 <= period_start_date.month <= 6):
            continue

        y7_price = q7.get('current_price', np.nan)
        y8_price = q8.get('target_price_next_q', np.nan)

        if pd.isna(y7_price):
            continue

        # Extract Q1-Q7 features
        feat_rows = seq.iloc[:7]
        X = feat_rows[feature_cols].copy()
        X = X.ffill().bfill().fillna(0.0)

        # Scale features
        X_scaled = (X.values - scaler_mean) / scaler_scale
        X_scaled = X_scaled.astype(np.float32)

        # Calculate days from quarter end to period start (transcript availability)
        fiscal_qtr_end = pd.to_datetime(q7.get('fiscal_quarter_end'))
        days_to_period_start = (period_start_date - fiscal_qtr_end).days

        sequences.append({
            'sequence_id': seq_id,
            'ticker': q7['ticker'],
            'period_start_date': period_start_date,
            'days_to_period_start': days_to_period_start,
            'X': X_scaled,
            'y7_price': float(y7_price),
            'y8_price_data': float(y8_price) if not pd.isna(y8_price) else None
        })

    logger.info(f"Found {len(sequences)} sequences for Q2 2025")
    return pd.DataFrame(sequences)


# ============================= Live Price Fetching ===========================
def fetch_q3_2025_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch actual stock prices as of Q3 2025 end (September 30, 2025)."""
    logger.info(f"Fetching Q3 2025 prices for {len(tickers)} tickers...")

    # Q3 2025 end date
    q3_end = "2025-09-30"

    prices = {}
    failed = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # Get historical data around Q3 end
            hist = stock.history(start="2025-09-25", end="2025-10-05")

            if not hist.empty:
                # Get closest price to Q3 end
                prices[ticker] = float(hist['Close'].iloc[-1])
                logger.info(f"  {ticker}: ${prices[ticker]:.2f}")
            else:
                logger.warning(f"  {ticker}: No data available")
                failed.append(ticker)
        except Exception as e:
            logger.warning(f"  {ticker}: Failed to fetch - {str(e)}")
            failed.append(ticker)

    if failed:
        logger.warning(f"Failed to fetch prices for {len(failed)} tickers: {failed}")

    return prices


# ============================= Prediction ====================================
def generate_predictions(model: nn.Module, sequences_df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Generate predictions for Q2 2025 sequences."""
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

            predictions.append({
                'ticker': row['ticker'],
                'period_start_date': row['period_start_date'],
                'days_to_period_start': row['days_to_period_start'],
                'q2_price': y7_price,
                'q3_price_predicted': y8_price_pred,
                'predicted_return_pct': predicted_return_pct,
                'q3_price_data': row['y8_price_data']
            })

    logger.info(f"Generated {len(predictions)} predictions")
    return pd.DataFrame(predictions)


# ============================= Backtesting ===================================
def backtest_threshold(
    predictions_df: pd.DataFrame,
    q3_actual_prices: Dict[str, float],
    threshold_pct: float,
    investment_per_stock: float = 1000.0
) -> Dict:
    """Backtest a single threshold for Q2 2025."""

    # Filter stocks predicted to gain > threshold
    high_confidence = predictions_df[predictions_df['predicted_return_pct'] > threshold_pct].copy()

    if len(high_confidence) == 0:
        return None

    # Add actual Q3 prices
    high_confidence['q3_price_actual'] = high_confidence['ticker'].map(q3_actual_prices)

    # Remove stocks where we couldn't fetch actual price
    valid = high_confidence.dropna(subset=['q3_price_actual']).copy()

    if len(valid) == 0:
        return None

    # Calculate actual returns
    valid['actual_return_pct'] = (valid['q3_price_actual'] - valid['q2_price']) / valid['q2_price']
    valid['investment'] = investment_per_stock
    valid['dollar_return'] = valid['investment'] * valid['actual_return_pct']
    valid['correct_prediction'] = valid['actual_return_pct'] > 0

    # Calculate metrics
    total_invested = valid['investment'].sum()
    total_return = valid['dollar_return'].sum()
    roi_pct = (total_return / total_invested * 100) if total_invested > 0 else 0

    # Calculate standard deviation of returns
    returns_pct = valid['actual_return_pct'].values * 100
    std_dev_pct = float(np.std(returns_pct, ddof=1)) if len(returns_pct) > 1 else 0.0

    correct_count = valid['correct_prediction'].sum()
    accuracy_pct = (correct_count / len(valid) * 100) if len(valid) > 0 else 0

    # Stock details
    stocks = []
    for _, row in valid.iterrows():
        stocks.append({
            'ticker': row['ticker'],
            'days_to_period_start': int(row['days_to_period_start']),
            'q2_price': float(row['q2_price']),
            'q3_price_predicted': float(row['q3_price_predicted']),
            'q3_price_actual': float(row['q3_price_actual']),
            'predicted_return_pct': float(row['predicted_return_pct'] * 100),
            'actual_return_pct': float(row['actual_return_pct'] * 100),
            'investment': float(row['investment']),
            'dollar_return': float(row['dollar_return']),
            'correct': bool(row['correct_prediction'])
        })

    return {
        'threshold_pct': float(threshold_pct * 100),
        'num_positions': len(valid),
        'total_invested': float(total_invested),
        'total_return': float(total_return),
        'roi_pct': float(roi_pct),
        'std_dev_pct': std_dev_pct,
        'accuracy_pct': float(accuracy_pct),
        'correct_predictions': int(correct_count),
        'stocks': sorted(stocks, key=lambda x: x['actual_return_pct'], reverse=True)
    }


# ============================= Main ==========================================
def main():
    parser = argparse.ArgumentParser(description="Backtest Q2 2025 quarter with actual Q3 prices")
    parser.add_argument("--test", type=str, default="data_pipeline/data/sequences_8q_test.parquet",
                        help="Path to test sequences parquet")
    parser.add_argument("--model", type=str, default="models/best_model.pt",
                        help="Path to trained model checkpoint (default: final_model_all_data.pt trained on all data)")
    parser.add_argument("--min-threshold", type=float, default=0.05,
                        help="Minimum threshold to test (default: 0.05 = 5%%)")
    parser.add_argument("--max-threshold", type=float, default=0.25,
                        help="Maximum threshold to test (default: 0.25 = 25%%)")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Step size for threshold sweep (default: 0.05 = 5%%)")
    parser.add_argument("--investment-per-stock", type=float, default=1000.0,
                        help="Fixed dollar amount to invest per stock (default: $1,000)")
    parser.add_argument("--output", type=str, default="backtest_q2_2025_results.json",
                        help="Output JSON file for results")
    args = parser.parse_args()

    # Check files exist
    test_path = Path(args.test)
    model_path = Path(args.model)
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    feature_cols = checkpoint['feature_cols']
    scaler_mean = checkpoint['scaler_mean']
    scaler_scale = checkpoint['scaler_scale']
    use_attn = checkpoint.get('use_attn', True)

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

    # Load Q2 2025 data
    sequences_df = load_q2_2025_sequences(
        str(test_path),
        feature_cols,
        scaler_mean,
        scaler_scale
    )

    if len(sequences_df) == 0:
        logger.error("No Q2 2025 sequences found in test data!")
        return

    # Generate predictions
    predictions_df = generate_predictions(model, sequences_df, device)

    # Fetch actual Q3 2025 prices
    tickers = predictions_df['ticker'].unique().tolist()
    q3_actual_prices = fetch_q3_2025_prices(tickers)

    if len(q3_actual_prices) == 0:
        logger.error("Could not fetch any Q3 2025 prices!")
        return

    # Sweep thresholds
    thresholds = np.arange(args.min_threshold, args.max_threshold + args.step/2, args.step)
    logger.info("="*70)
    logger.info(f"TESTING Q2 2025 WITH THRESHOLDS {args.min_threshold*100:.0f}% TO {args.max_threshold*100:.0f}%")
    logger.info(f"Investment per stock: ${args.investment_per_stock:,.0f}")
    logger.info("="*70)

    results = []
    for threshold in thresholds:
        logger.info(f"\nTesting threshold: {threshold*100:.0f}%...")
        result = backtest_threshold(predictions_df, q3_actual_prices, threshold, args.investment_per_stock)
        if result is not None:
            results.append(result)
            logger.info(f"  ✓ ROI: {result['roi_pct']:.2f}%, Std Dev: {result['std_dev_pct']:.2f}%, Positions: {result['num_positions']}, Accuracy: {result['accuracy_pct']:.1f}%")
        else:
            logger.info(f"  ✗ No qualifying trades")

    # Print detailed results
    if results:
        best_result = max(results, key=lambda x: x['roi_pct'])

        logger.info("\n" + "="*70)
        logger.info("Q2 2025 BACKTEST COMPLETE")
        logger.info("="*70)
        logger.info(f"BEST THRESHOLD: {best_result['threshold_pct']:.0f}%")
        logger.info(f"  ROI: {best_result['roi_pct']:.2f}%")
        logger.info(f"  Std Dev: {best_result['std_dev_pct']:.2f}%")
        logger.info(f"  Positions: {best_result['num_positions']}")
        logger.info(f"  Accuracy: {best_result['accuracy_pct']:.1f}%")
        logger.info("="*70)

        # Print stock details for best threshold
        print("\n" + "="*135)
        print(f"STOCKS SELECTED AT {best_result['threshold_pct']:.0f}% THRESHOLD (Company-Specific Rebalance Dates)")
        print("="*135)
        print(f"{'Ticker':<8} {'Days':>5} {'Q2 Price':>10} {'Q3 Pred':>10} {'Q3 Actual':>10} {'Pred %':>8} {'Actual %':>9} {'Invested':>10} {'Return':>10} {'Correct':<8}")
        print(f"{'':8} {'Lag':>5} {'':<10} {'':<10} {'':<10} {'':<8} {'':<9} {'':<10} {'':<10} {'':<8}")
        print("-"*135)

        for stock in best_result['stocks']:
            correct_mark = "YES" if stock['correct'] else "NO "
            print(f"{stock['ticker']:<8} {stock['days_to_period_start']:>5d} ${stock['q2_price']:>9.2f} ${stock['q3_price_predicted']:>9.2f} ${stock['q3_price_actual']:>9.2f} "
                  f"{stock['predicted_return_pct']:>7.1f}% {stock['actual_return_pct']:>8.1f}% "
                  f"${stock['investment']:>9.0f} ${stock['dollar_return']:>9.2f} {correct_mark:<8}")

        print("-"*135)
        print(f"{'TOTAL':<8} {'':<5} {'':<10} {'':<10} {'':<10} {'':<8} {'':<9} ${best_result['total_invested']:>9.0f} ${best_result['total_return']:>9.2f}")
        print("="*135)
        print(f"ROI: {best_result['roi_pct']:.2f}% | Accuracy: {best_result['correct_predictions']}/{best_result['num_positions']} ({best_result['accuracy_pct']:.1f}%)")
        print(f"Note: 'Days Lag' shows company-specific days from Q2 end to rebalance date based on historical transcript timing")
        print("="*135)

        # Save results
        output_data = {
            'quarter': 'Q2 2025',
            'analysis_date': datetime.now().isoformat(),
            'sweep_parameters': {
                'min_threshold': args.min_threshold * 100,
                'max_threshold': args.max_threshold * 100,
                'step': args.step * 100,
                'investment_per_stock': args.investment_per_stock
            },
            'best_threshold': best_result,
            'all_results': results
        }

        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")

        # Summary table
        print("\n" + "="*110)
        print("THRESHOLD COMPARISON")
        print("="*110)
        print(f"{'Threshold':>10} {'Positions':>10} {'ROI %':>10} {'Std Dev %':>12} {'Accuracy %':>12} {'Total Return':>15}")
        print("-"*110)
        for r in sorted(results, key=lambda x: x['threshold_pct']):
            print(f"{r['threshold_pct']:>9.0f}% {r['num_positions']:>10} {r['roi_pct']:>9.2f}% {r['std_dev_pct']:>11.2f}% {r['accuracy_pct']:>11.1f}% ${r['total_return']:>14.2f}")
        print("="*110)
    else:
        logger.error("No valid results found!")


if __name__ == "__main__":
    main()

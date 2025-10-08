"""
Backtest LSTM No-Price Model: Quarterly Trading Strategy
==========================================================

Strategy:
1. At each quarter, invest $100 in stocks predicted to go UP (long position)
2. Short $100 worth of stocks predicted to go DOWN
3. Hold positions for the quarter, then liquidate and rebalance
4. Track cumulative returns over the entire 20-year test period

This script loads the trained LSTM model, generates predictions for all test sequences,
and simulates the trading strategy to calculate average quarterly returns.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_lstm_no_price")

# ============================= Constants =====================================
BASE_EXCLUDE_COLS = [
    "sequence_id", "ticker", "quarter_in_sequence", "sequence_start_date",
    "sequence_end_date", "fiscal_quarter_end", "sector", "year",
    "transcript_date", "transcript_type", "days_after_quarter",
    "target_price_next_q", "current_price", "rebalance_date", "in_sp500"
]

PRICE_DERIVED_COLS = [
    "market_cap", "enterprise_value", "pe_ratio", "earnings_yield", "momentum_12m"
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


class UniLSTMAttnDirect(nn.Module):
    """LSTM that predicts Q8 price DIRECTLY (no Q7 price input)."""
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
        self.fc1 = nn.Linear(rep_dim, 128)
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
def load_sequences_and_prepare(
    test_path: str,
    feature_cols: List[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray
) -> pd.DataFrame:
    """
    Load test sequences and prepare features for prediction.

    Returns:
        DataFrame with columns: sequence_id, ticker, rebalance_date, X_features (7xF),
                                y7_price, y8_price_actual
    """
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

        # Extract Q1-Q7 features (first 7 quarters)
        feat_rows = seq.iloc[:7]
        X = feat_rows[feature_cols].copy()
        X = X.ffill().bfill().fillna(0.0)

        # Scale features (Q1-Q7 only)
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
    """
    Generate predictions for all sequences.

    Returns:
        DataFrame with columns: sequence_id, ticker, rebalance_date, y7_price, y8_price_actual,
                                y8_price_predicted, predicted_direction (1=up, 0=down)
    """
    logger.info("Generating predictions...")
    model.eval()

    predictions = []
    with torch.no_grad():
        for idx, row in sequences_df.iterrows():
            X = torch.from_numpy(row['X']).unsqueeze(0).to(device)  # (1, 7, F)
            y8_log_pred = model(X)  # (1, 1)
            y8_price_pred = float(torch.expm1(y8_log_pred).cpu().item())

            predicted_direction = 1 if y8_price_pred > row['y7_price'] else 0

            predictions.append({
                'sequence_id': row['sequence_id'],
                'ticker': row['ticker'],
                'rebalance_date': row['rebalance_date'],
                'y7_price': row['y7_price'],
                'y8_price_actual': row['y8_price_actual'],
                'y8_price_predicted': y8_price_pred,
                'predicted_direction': predicted_direction,
                'actual_direction': 1 if row['y8_price_actual'] > row['y7_price'] else 0,
                'actual_return': (row['y8_price_actual'] - row['y7_price']) / row['y7_price']
            })

    logger.info(f"Generated {len(predictions)} predictions")
    return pd.DataFrame(predictions)


# ============================= Backtesting ===================================
def backtest_strategy(predictions_df: pd.DataFrame, initial_investment: float = 100.0) -> Dict:
    """
    Backtest the trading strategy: $100 long on predicted-UP, $100 short on predicted-DOWN.

    Strategy:
    - Each quarter, for each stock predicted to go UP: invest $100 (long)
    - Each quarter, for each stock predicted to go DOWN: short $100 worth
    - Calculate returns at end of quarter based on actual price movement
    - Track cumulative returns across all quarters

    Returns:
        Dictionary with backtest results and statistics
    """
    logger.info("="*70)
    logger.info("BACKTESTING QUARTERLY REBALANCING STRATEGY")
    logger.info("="*70)

    # Sort by rebalance date to simulate chronological trading
    predictions_df['rebalance_date'] = pd.to_datetime(predictions_df['rebalance_date'])
    predictions_df = predictions_df.sort_values('rebalance_date')

    # Group by quarter
    quarterly_results = []

    for quarter_date, quarter_group in predictions_df.groupby('rebalance_date'):
        # Split into UP and DOWN predictions
        up_predictions = quarter_group[quarter_group['predicted_direction'] == 1]
        down_predictions = quarter_group[quarter_group['predicted_direction'] == 0]

        # Calculate returns for long positions (predicted UP)
        long_returns = []
        for _, row in up_predictions.iterrows():
            actual_return = row['actual_return']
            dollar_return = initial_investment * actual_return
            long_returns.append({
                'ticker': row['ticker'],
                'position': 'LONG',
                'predicted_direction': 'UP',
                'actual_direction': 'UP' if row['actual_direction'] == 1 else 'DOWN',
                'return_pct': actual_return * 100,
                'dollar_return': dollar_return
            })

        # Calculate returns for short positions (predicted DOWN)
        short_returns = []
        for _, row in down_predictions.iterrows():
            actual_return = row['actual_return']
            # For shorts: profit when price goes down, loss when price goes up
            dollar_return = -initial_investment * actual_return
            short_returns.append({
                'ticker': row['ticker'],
                'position': 'SHORT',
                'predicted_direction': 'DOWN',
                'actual_direction': 'UP' if row['actual_direction'] == 1 else 'DOWN',
                'return_pct': -actual_return * 100,
                'dollar_return': dollar_return
            })

        all_trades = long_returns + short_returns
        total_return = sum(t['dollar_return'] for t in all_trades)
        long_total_return = sum(t['dollar_return'] for t in long_returns)
        short_total_return = sum(t['dollar_return'] for t in short_returns)
        num_trades = len(all_trades)
        avg_return_per_trade = total_return / num_trades if num_trades > 0 else 0

        quarterly_results.append({
            'quarter_date': quarter_date,
            'num_long_positions': len(long_returns),
            'num_short_positions': len(short_returns),
            'total_trades': num_trades,
            'total_dollar_return': total_return,
            'long_dollar_return': long_total_return,
            'short_dollar_return': short_total_return,
            'avg_return_per_trade': avg_return_per_trade,
            'total_invested': initial_investment * num_trades,
            'long_invested': initial_investment * len(long_returns),
            'short_invested': initial_investment * len(short_returns),
            'trades': all_trades
        })

    # Calculate overall statistics
    total_quarters = len(quarterly_results)
    total_trades_all = sum(q['total_trades'] for q in quarterly_results)
    total_dollar_returns = sum(q['total_dollar_return'] for q in quarterly_results)
    total_long_returns = sum(q['long_dollar_return'] for q in quarterly_results)
    total_short_returns = sum(q['short_dollar_return'] for q in quarterly_results)
    total_invested_all = sum(q['total_invested'] for q in quarterly_results)
    total_long_invested = sum(q['long_invested'] for q in quarterly_results)
    total_short_invested = sum(q['short_invested'] for q in quarterly_results)

    avg_return_per_quarter = total_dollar_returns / total_quarters if total_quarters > 0 else 0
    avg_return_per_trade = total_dollar_returns / total_trades_all if total_trades_all > 0 else 0
    overall_roi = (total_dollar_returns / total_invested_all * 100) if total_invested_all > 0 else 0
    long_roi = (total_long_returns / total_long_invested * 100) if total_long_invested > 0 else 0
    short_roi = (total_short_returns / total_short_invested * 100) if total_short_invested > 0 else 0

    # Calculate accuracy metrics
    all_trades = [trade for q in quarterly_results for trade in q['trades']]
    correct_long = sum(1 for t in all_trades if t['position'] == 'LONG' and t['actual_direction'] == 'UP')
    total_long = sum(1 for t in all_trades if t['position'] == 'LONG')
    correct_short = sum(1 for t in all_trades if t['position'] == 'SHORT' and t['actual_direction'] == 'DOWN')
    total_short = sum(1 for t in all_trades if t['position'] == 'SHORT')

    long_accuracy = correct_long / total_long * 100 if total_long > 0 else 0
    short_accuracy = correct_short / total_short * 100 if total_short > 0 else 0
    overall_accuracy = (correct_long + correct_short) / len(all_trades) * 100 if len(all_trades) > 0 else 0

    # Log summary
    logger.info(f"\nBACKTEST PERIOD: {quarterly_results[0]['quarter_date']} to {quarterly_results[-1]['quarter_date']}")
    logger.info(f"Total Quarters: {total_quarters}")
    logger.info(f"Total Trades: {total_trades_all:,}")
    logger.info(f"  - Long Positions: {total_long:,}")
    logger.info(f"  - Short Positions: {total_short:,}")
    logger.info(f"\nDIRECTIONAL ACCURACY:")
    logger.info(f"  Overall: {overall_accuracy:.2f}%")
    logger.info(f"  Long Positions: {long_accuracy:.2f}% ({correct_long}/{total_long})")
    logger.info(f"  Short Positions: {short_accuracy:.2f}% ({correct_short}/{total_short})")
    logger.info(f"\nFINANCIAL PERFORMANCE:")
    logger.info(f"  Total Invested: ${total_invested_all:,.2f}")
    logger.info(f"  Total Returns: ${total_dollar_returns:,.2f}")
    logger.info(f"  Average Return per Quarter: ${avg_return_per_quarter:,.2f}")
    logger.info(f"  Average Return per Trade: ${avg_return_per_trade:.2f}")
    logger.info(f"  Overall ROI: {overall_roi:.2f}%")
    logger.info(f"\nROI BY STRATEGY:")
    logger.info(f"  Long Positions ROI: {long_roi:.2f}% (${total_long_returns:,.2f} / ${total_long_invested:,.2f})")
    logger.info(f"  Short Positions ROI: {short_roi:.2f}% (${total_short_returns:,.2f} / ${total_short_invested:,.2f})")
    logger.info("="*70)

    return {
        'summary': {
            'total_quarters': total_quarters,
            'total_trades': total_trades_all,
            'total_long_positions': total_long,
            'total_short_positions': total_short,
            'total_invested': float(total_invested_all),
            'total_returns': float(total_dollar_returns),
            'avg_return_per_quarter': float(avg_return_per_quarter),
            'avg_return_per_trade': float(avg_return_per_trade),
            'overall_roi_pct': float(overall_roi),
            'long_positions': {
                'total_invested': float(total_long_invested),
                'total_returns': float(total_long_returns),
                'roi_pct': float(long_roi)
            },
            'short_positions': {
                'total_invested': float(total_short_invested),
                'total_returns': float(total_short_returns),
                'roi_pct': float(short_roi)
            },
            'directional_accuracy': {
                'overall_pct': float(overall_accuracy),
                'long_pct': float(long_accuracy),
                'short_pct': float(short_accuracy),
                'correct_long': int(correct_long),
                'correct_short': int(correct_short)
            }
        },
        'quarterly_results': [
            {
                'quarter_date': str(q['quarter_date']),
                'num_long_positions': q['num_long_positions'],
                'num_short_positions': q['num_short_positions'],
                'total_trades': q['total_trades'],
                'total_dollar_return': float(q['total_dollar_return']),
                'avg_return_per_trade': float(q['avg_return_per_trade']),
                'total_invested': float(q['total_invested'])
            }
            for q in quarterly_results
        ]
    }


# ============================= Main ==========================================
def main():
    parser = argparse.ArgumentParser(description="Backtest LSTM No-Price quarterly trading strategy")
    parser.add_argument("--test", type=str, default="data_pipeline/data/sequences_8q_test.parquet",
                        help="Path to test sequences parquet")
    parser.add_argument("--model", type=str, default="models_no_price/best_model_no_price.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--investment", type=float, default=100.0,
                        help="Dollar amount to invest per position (default: $100)")
    parser.add_argument("--output", type=str, default="backtest_results_lstm_no_price.json",
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
    model = UniLSTMAttnDirect(
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
    backtest_results = backtest_strategy(predictions_df, initial_investment=args.investment)

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(backtest_results, f, indent=2)
    logger.info(f"\nBacktest results saved to {output_path}")

    # Print final summary
    summary = backtest_results['summary']
    print("\n" + "="*70)
    print("FINAL BACKTEST SUMMARY")
    print("="*70)
    print(f"Total Quarters Traded: {summary['total_quarters']}")
    print(f"Total Trades Executed: {summary['total_trades']:,}")
    print(f"Total Capital Deployed: ${summary['total_invested']:,.2f}")
    print(f"Total Returns: ${summary['total_returns']:,.2f}")
    print(f"Average Return per Quarter: ${summary['avg_return_per_quarter']:,.2f}")
    print(f"Average Return per $100 Trade: ${summary['avg_return_per_trade']:.2f}")
    print(f"Overall ROI: {summary['overall_roi_pct']:.2f}%")
    print(f"\nROI BY STRATEGY:")
    print(f"  LONG Positions ROI: {summary['long_positions']['roi_pct']:.2f}%")
    print(f"    (${summary['long_positions']['total_returns']:,.2f} / ${summary['long_positions']['total_invested']:,.2f})")
    print(f"  SHORT Positions ROI: {summary['short_positions']['roi_pct']:.2f}%")
    print(f"    (${summary['short_positions']['total_returns']:,.2f} / ${summary['short_positions']['total_invested']:,.2f})")
    print(f"\nDirectional Accuracy: {summary['directional_accuracy']['overall_pct']:.2f}%")
    print(f"  Long Accuracy: {summary['directional_accuracy']['long_pct']:.2f}%")
    print(f"  Short Accuracy: {summary['directional_accuracy']['short_pct']:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()

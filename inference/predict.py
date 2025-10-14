
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ----------------------------- Logging ---------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("predict")

# ----------------------------- Model (from train_lstm.py) ---------------------------
class CausalAttention(torch.nn.Module):
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

class UniLSTMAttnDelta(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 192, num_layers: int = 2, dropout: float = 0.25, use_attn: bool = True):
        super().__init__()
        self.use_attn = use_attn
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attn = CausalAttention(hidden_size) if use_attn else None
        rep_dim = hidden_size * (2 if use_attn else 1)
        self.ln = torch.nn.LayerNorm(rep_dim)
        self.fc1 = torch.nn.Linear(rep_dim + 1, 128)
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(128, 1)

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
        delta_log = self.fc2(z)
        y8_log_hat = y7_log + delta_log
        return y8_log_hat, delta_log

# ----------------------------- Predictor ----------------------------------
class Predictor:
    def __init__(self, model_path: str, data_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model_path = Path(model_path)
        self.data_path = Path(data_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.sequences_df = pd.read_parquet(self.data_path)
        self.sequences_df['fiscal_quarter_end'] = pd.to_datetime(self.sequences_df['fiscal_quarter_end'])

        self.model, self.feature_cols, self.scaler = self._load_model_and_scaler()

    def _load_model_and_scaler(self):
        logger.info(f"Loading model from {self.model_path}")
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Recreate model architecture from checkpoint
        model = UniLSTMAttnDelta(
            input_size=len(ckpt["feature_cols"]),
            use_attn=ckpt.get("use_attn", True)
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()

        # Load scaler
        scaler = StandardScaler()
        scaler.mean_ = ckpt["scaler_mean"]
        scaler.scale_ = ckpt["scaler_scale"]

        logger.info(f"Model loaded. Features: {len(ckpt['feature_cols'])}")
        return model, ckpt["feature_cols"], scaler

    def predict_latest(self, ticker: str):
        logger.info(f"Finding latest available sequence for {ticker} to make a prediction.")

        # Find all sequences for the given ticker
        ticker_sequences = self.sequences_df[self.sequences_df['ticker'] == ticker]
        if ticker_sequences.empty:
            logger.error(f"No data found for ticker: {ticker}")
            return None

        # Find the most recent 7th quarter for the ticker
        latest_q7 = ticker_sequences[ticker_sequences['quarter_in_sequence'] == 6].sort_values('fiscal_quarter_end', ascending=False).iloc[0]
        sequence_id = latest_q7['sequence_id']
        
        q7_date_str = latest_q7['fiscal_quarter_end'].strftime('%Y-%m-%d')
        logger.info(f"Found latest input sequence, ending on: {q7_date_str} (sequence_id: {sequence_id})")

        # Get the full 8-quarter sequence
        full_sequence = self.sequences_df[self.sequences_df['sequence_id'] == sequence_id].sort_values("quarter_in_sequence")

        if len(full_sequence) != 8:
            logger.error(f"Sequence {sequence_id} does not have 8 quarters. Found {len(full_sequence)}.")
            return None

        # Prepare the input data (first 7 quarters)
        input_df = full_sequence.iloc[:7].copy()
        
        # Scale features
        input_df[self.feature_cols] = self.scaler.transform(input_df[self.feature_cols])
        
        X = input_df[self.feature_cols].ffill().bfill().fillna(0.0).values.astype(np.float32)
        X_tensor = torch.from_numpy(X).unsqueeze(0).to(self.device) # Add batch dimension

        # Get y7_log (current_price of the 7th quarter)
        y7_raw = full_sequence.iloc[6]['current_price']

        if pd.isna(y7_raw):
            q7_date_str = full_sequence.iloc[6]['fiscal_quarter_end'].strftime('%Y-%m-%d')
            logger.error(f"Cannot make a prediction for {ticker}.")
            logger.error(f"The required input price ('current_price') for the last quarter of the input sequence (ending {q7_date_str}) is missing (NaN).")
            logger.error("This is a data quality issue. To get a prediction, the data source needs to be fixed so this value is present.")
            return None

        y7_log = np.log1p(float(y7_raw))
        y7_log_tensor = torch.tensor([[y7_log]], dtype=torch.float32).to(self.device)

        # Run prediction
        with torch.no_grad():
            y8_log_hat, _ = self.model(X_tensor, y7_log_tensor)

        # Inverse transform the prediction
        predicted_log_price = y8_log_hat.item()
        predicted_price = np.expm1(predicted_log_price)

        # Get actual target price for comparison if it exists
        actual_price = full_sequence.iloc[7]['target_price_next_q']

        q7_date = full_sequence.iloc[6]['fiscal_quarter_end'].strftime('%Y-%m-%d')
        q8_date = full_sequence.iloc[7]['fiscal_quarter_end'].strftime('%Y-%m-%d')

        result = {
            "ticker": ticker,
            "prediction_for_quarter_ending": q8_date,
            "input_sequence_ends_on": q7_date,
            "last_known_price": float(y7_raw),
            "predicted_price": float(predicted_price),
            "actual_price": float(actual_price) if pd.notna(actual_price) else "N/A",
        }
        
        return result

def main():
    parser = argparse.ArgumentParser(
        description="Inference script for the LSTM model. Predicts the next quarter's price for a given ticker based on the latest available data.",
        epilog="Note: This script uses pre-processed data. To make predictions on the very latest market data, please re-run the full data pipeline first by executing 'python data_pipeline/run_pipeline.py'."
    )
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol to predict for (e.g., 'AAPL').")
    parser.add_argument("--model-path", type=str, default="models/final_model_all_data.pt", help="Path to the trained model checkpoint (default: final model trained on all data).")
    parser.add_argument("--data-path", type=str, default="data_pipeline/data/sequences_8q.parquet", help="Path to the sequences data file.")
    
    args = parser.parse_args()

    try:
        predictor = Predictor(model_path=args.model_path, data_path=args.data_path)
        prediction = predictor.predict_latest(ticker=args.ticker)

        if prediction:
            logger.info("="*60)
            logger.info("PREDICTION COMPLETE")
            logger.info("="*60)
            print(json.dumps(prediction, indent=4))
            logger.info("="*60)

            # Provide interpretation
            last_price = prediction['last_known_price']
            pred_price = prediction['predicted_price']
            change = pred_price - last_price
            percent_change = (change / last_price) * 100 if last_price != 0 else float('inf')

            logger.info("Interpretation:")
            logger.info(f"The model predicts the share price for {prediction['ticker']} will be {pred_price:.2f} for the quarter ending {prediction['prediction_for_quarter_ending']}.")
            logger.info(f"This is a change of {change:+.2f} ({percent_change:+.2f}%) from the last known value of {last_price:.2f} on {prediction['input_sequence_ends_on']}.")
            if prediction['actual_price'] != "N/A":
                 logger.info(f"For reference, the actual value was {prediction['actual_price']:.2f}.")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()

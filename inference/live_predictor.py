#!/usr/bin/env python3
"""
Live Stock Price Predictor using existing processed data
Uses the most recent 7 quarters from quarters.parquet to predict the next quarter
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LivePredictor:
    """Use existing processed data to make predictions for the next quarter"""

    def __init__(self, data_path: str = "data_pipeline/data/quarters.parquet"):
        self.data_path = Path(data_path)
        self.quarters_needed = 7
        self.quarters_df = None
        self._load_data()

    def _load_data(self):
        """Load the processed quarters data"""
        try:
            logger.info(f"Loading processed data from {self.data_path}")
            self.quarters_df = pd.read_parquet(self.data_path)
            self.quarters_df['fiscal_quarter_end'] = pd.to_datetime(self.quarters_df['fiscal_quarter_end'])
            logger.info(f"Loaded {len(self.quarters_df)} quarters from {self.quarters_df['fiscal_quarter_end'].min()} to {self.quarters_df['fiscal_quarter_end'].max()}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def get_recent_quarters_for_ticker(self, ticker: str):
        """Get the most recent 7 quarters for a ticker"""
        ticker_data = self.quarters_df[self.quarters_df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('fiscal_quarter_end')

        if len(ticker_data) < 7:
            logger.warning(f"Only {len(ticker_data)} quarters available for {ticker}, need 7")
            return None

        recent_7 = ticker_data.tail(7)
        return recent_7

    def prepare_for_model(self, quarters_df: pd.DataFrame, feature_cols: list, scaler):
        """Prepare the 7-quarter sequence for model input"""
        try:
            available_features = [col for col in feature_cols if col in quarters_df.columns]
            missing_features = [col for col in feature_cols if col not in quarters_df.columns]

            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features")

            feature_data = pd.DataFrame(index=quarters_df.index)
            for col in feature_cols:
                if col in quarters_df.columns:
                    feature_data[col] = quarters_df[col]
                else:
                    feature_data[col] = 0

            feature_data = feature_data.ffill().bfill().fillna(0.0)
            scaled_features = scaler.transform(feature_data)
            current_price = quarters_df.iloc[-1]['current_price']

            return scaled_features, current_price, len(available_features)
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None, None, 0

    def predict_next_quarter(self, ticker: str, model, feature_cols: list, scaler, device):
        """Make a prediction for the next quarter using existing processed data"""
        import torch

        logger.info(f"Making live prediction for {ticker} using processed data...")

        quarters_df = self.get_recent_quarters_for_ticker(ticker)
        if quarters_df is None:
            return {'success': False, 'error': f'Insufficient data for {ticker}'}

        quarter_dates = quarters_df['fiscal_quarter_end'].dt.strftime('%Y-%m-%d').tolist()
        logger.info(f"Using quarters: {quarter_dates}")

        X_scaled, current_price, features_available = self.prepare_for_model(quarters_df, feature_cols, scaler)
        if X_scaled is None:
            return {'success': False, 'error': 'Could not prepare features'}

        X_tensor = torch.from_numpy(X_scaled.astype(np.float32)).unsqueeze(0).to(device)
        y7_log = np.log1p(current_price)
        y7_log_tensor = torch.tensor([[y7_log]], dtype=torch.float32).to(device)

        with torch.no_grad():
            y8_log_hat, delta_log, attn_weights = model(X_tensor, y7_log_tensor)

        predicted_price = np.expm1(y8_log_hat.item())
        last_quarter = quarters_df.iloc[-1]['fiscal_quarter_end']
        next_quarter = (pd.Period(last_quarter, freq='Q') + 1).end_time

        return {
            'success': True,
            'ticker': ticker,
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'price_change': float(predicted_price - current_price),
            'price_change_pct': float((predicted_price - current_price) / current_price * 100),
            'delta_log': delta_log.item(),
            'input_sequence_ends': last_quarter.strftime('%Y-%m-%d'),
            'prediction_for_quarter': next_quarter.strftime('%Y-%m-%d'),
            'attn_weights': attn_weights[0].cpu().numpy() if attn_weights is not None else None,
            'quarters_used': quarter_dates,
            'features_available': features_available,
            'features_total': len(feature_cols)
        }

#!/usr/bin/env python3
"""
Live Stock Price Predictor using real-time data
Fetches the most recent 7 quarters live from FMP API for predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import os
import requests
from live_data_fetcher import LiveDataFetcher
from live_feature_engineering import LiveFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LivePredictor:
    """Fetch live data and make predictions for the next quarter"""

    def __init__(self, use_cached_data: bool = False, use_transcript_embeddings: bool = True, data_path: str = "data_pipeline/data/quarters.parquet"):
        self.quarters_needed = 7
        self.use_cached_data = use_cached_data
        self.use_transcript_embeddings = use_transcript_embeddings
        self.data_path = Path(data_path)
        self.quarters_df = None
        self.data_fetcher = LiveDataFetcher()
        self.feature_engineer = LiveFeatureEngineer(use_transcript_embeddings=use_transcript_embeddings)

        if use_cached_data:
            self._load_cached_data()

    def _load_cached_data(self):
        """Load the processed quarters data from cache (old method)"""
        try:
            logger.info(f"Loading cached data from {self.data_path}")
            self.quarters_df = pd.read_parquet(self.data_path)
            self.quarters_df['fiscal_quarter_end'] = pd.to_datetime(self.quarters_df['fiscal_quarter_end'])
            logger.info(f"Loaded {len(self.quarters_df)} quarters from {self.quarters_df['fiscal_quarter_end'].min()} to {self.quarters_df['fiscal_quarter_end'].max()}")
        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")
            raise

    def get_eligible_tickers(self):
        """Get all S&P 500 tickers that can have live predictions"""
        # If using cached data, filter from cache
        if self.use_cached_data and self.quarters_df is not None:
            eligible_tickers = []

            for ticker in self.quarters_df['ticker'].unique():
                ticker_data = self.quarters_df[self.quarters_df['ticker'] == ticker].sort_values('fiscal_quarter_end')

                # Check if ticker has at least 7 quarters
                if len(ticker_data) < 7:
                    continue

                # Get most recent 7 quarters
                recent_7 = ticker_data.tail(7)

                # Check if all 7 quarters have in_sp500 = True
                if not recent_7['in_sp500'].all():
                    continue

                # Check if the most recent quarter has transcript data (period_start_date is not null)
                if 'period_start_date' in recent_7.columns:
                    most_recent_transcript_date = pd.to_datetime(recent_7.iloc[-1]['period_start_date'])
                    if pd.notna(most_recent_transcript_date):
                        eligible_tickers.append(ticker)

            logger.info(f"Found {len(eligible_tickers)} S&P 500 companies with complete 7-quarter sequence in cache")
            return sorted(eligible_tickers)

        # If using live data, fetch S&P 500 list
        else:
            try:
                # Fetch S&P 500 constituent list
                url = f"https://financialmodelingprep.com/api/v3/sp500_constituent"
                params = {'apikey': self.data_fetcher.api_key}
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                sp500_data = response.json()

                tickers = [item['symbol'] for item in sp500_data]
                logger.info(f"Found {len(tickers)} S&P 500 companies available for live predictions")
                return sorted(tickers)

            except Exception as e:
                logger.error(f"Error fetching S&P 500 list: {e}")
                return []

    def get_recent_quarters_for_ticker(self, ticker: str):
        """Get the most recent 7 quarters for a ticker (live or cached)"""
        # If using live data, fetch fresh data
        if not self.use_cached_data:
            logger.info(f"Fetching live data for {ticker}...")
            quarters_df = self.data_fetcher.fetch_complete_data(ticker, quarters=7)

            if quarters_df is None or len(quarters_df) < 7:
                logger.warning(f"Could not fetch complete data for {ticker}")
                return None

            # Engineer features for live data
            logger.info(f"Engineering features for {ticker}...")
            quarters_df = self.feature_engineer.engineer_features(quarters_df)

            return quarters_df

        # If using cached data (already has features)
        else:
            ticker_data = self.quarters_df[self.quarters_df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('fiscal_quarter_end')

            if len(ticker_data) < 7:
                logger.warning(f"Only {len(ticker_data)} quarters available for {ticker}, need 7")
                return None

            recent_7 = ticker_data.tail(7)

            # Verify all 7 quarters are in S&P 500
            if not recent_7['in_sp500'].all():
                logger.warning(f"{ticker} has not been in S&P 500 for all of the past 7 quarters")
                return None

            return recent_7

    def get_current_price(self, ticker: str):
        """Fetch current (live) stock price using FMP API"""
        try:
            api_key = os.environ.get('FMP_API_KEY', 'FlVMN0WC4KpV1f9OWTNVamh39gW8wx7L')

            # FMP real-time quote endpoint
            url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}"
            params = {'apikey': api_key}

            logger.info(f"Fetching current price for {ticker}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data and len(data) > 0:
                current_price = data[0]['price']
                logger.info(f"Current price for {ticker}: ${current_price:.2f}")
                return float(current_price)
            else:
                logger.warning(f"No current price data available for {ticker}")
                return None

        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {e}")
            return None

    def get_actual_price_for_date(self, ticker: str, target_date: pd.Timestamp):
        """Fetch actual stock price for a specific date using FMP API"""
        try:
            api_key = os.environ.get('FMP_API_KEY', 'FlVMN0WC4KpV1f9OWTNVamh39gW8wx7L')

            # Format date as YYYY-MM-DD
            date_str = target_date.strftime('%Y-%m-%d')

            # FMP historical price endpoint - gets daily prices
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
            params = {
                'from': date_str,
                'to': date_str,
                'apikey': api_key
            }

            logger.info(f"Fetching actual price for {ticker} on {date_str}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'historical' in data and len(data['historical']) > 0:
                actual_price = data['historical'][0]['close']
                logger.info(f"Found actual price for {ticker} on {date_str}: ${actual_price:.2f}")
                return float(actual_price)
            else:
                # Try to get the nearest trading day (within 5 days)
                logger.warning(f"No price data for {ticker} on exact date {date_str}, trying nearby dates")

                start_date = (target_date - timedelta(days=5)).strftime('%Y-%m-%d')
                end_date = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')

                params['from'] = start_date
                params['to'] = end_date

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if 'historical' in data and len(data['historical']) > 0:
                    # Get the closest date
                    historical = pd.DataFrame(data['historical'])
                    historical['date'] = pd.to_datetime(historical['date'])
                    historical['days_diff'] = abs((historical['date'] - target_date).dt.days)
                    closest = historical.loc[historical['days_diff'].idxmin()]

                    actual_price = float(closest['close'])
                    actual_date = closest['date'].strftime('%Y-%m-%d')
                    logger.info(f"Found closest price for {ticker} on {actual_date}: ${actual_price:.2f} ({closest['days_diff']:.0f} days from target)")
                    return actual_price
                else:
                    logger.warning(f"No historical price data available for {ticker} near {date_str}")
                    return None

        except Exception as e:
            logger.error(f"Error fetching actual price for {ticker}: {e}")
            return None

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
        last_quarter_fiscal_end = quarters_df.iloc[-1]['fiscal_quarter_end']

        # Calculate company-specific prediction target date based on most recent holding period
        # The prediction target is when the NEXT quarter's transcript is expected to be released
        # This is estimated by adding the last holding period to the most recent transcript release date
        if 'period_start_date' in quarters_df.columns and 'holding_period_days' in quarters_df.columns:
            # Get the most recent quarter's period_start_date (when Q7 transcript was released)
            last_period_start = pd.to_datetime(quarters_df.iloc[-1]['period_start_date'])

            # Get the most recent holding period length (company-specific)
            last_holding_period = quarters_df.iloc[-1]['holding_period_days']

            # Predicted next transcript release date = last transcript release + last holding period
            prediction_target_date = last_period_start + pd.Timedelta(days=last_holding_period)

            logger.info(f"{ticker} most recent fiscal quarter end: {last_quarter_fiscal_end.strftime('%Y-%m-%d')}")
            logger.info(f"{ticker} most recent transcript released: {last_period_start.strftime('%Y-%m-%d')}")
            logger.info(f"{ticker} last holding period: {last_holding_period:.0f} days")
            logger.info(f"Predicted next transcript availability: {prediction_target_date.strftime('%Y-%m-%d')}")

            # Input sequence ends when the most recent transcript was released (period_start_date)
            input_sequence_end_date = last_period_start
        else:
            # Fallback to generic quarterly calendar
            next_quarter_fiscal_end = (pd.Period(last_quarter_fiscal_end, freq='Q') + 1).end_time
            prediction_target_date = next_quarter_fiscal_end
            input_sequence_end_date = last_quarter_fiscal_end
            logger.warning(f"holding_period_days not available, using generic quarterly calendar")

        # Get the CURRENT live stock price (not the stored historical price)
        live_current_price = self.get_current_price(ticker)
        if live_current_price is None:
            logger.warning(f"Could not fetch current price for {ticker}, using stored historical price")
            live_current_price = current_price

        # Get the price when the most recent transcript was released (quarter start/period start)
        quarter_start_price = self.get_actual_price_for_date(ticker, input_sequence_end_date)
        if quarter_start_price is None:
            logger.warning(f"Could not fetch quarter start price, using quarter end price")
            quarter_start_price = current_price

        # Check if we're making a truly forward prediction or if target date has passed
        is_future_prediction = prediction_target_date > pd.Timestamp.now()

        result = {
            'success': True,
            'ticker': ticker,
            'current_price': float(live_current_price),
            'quarter_start_price': float(quarter_start_price),  # Price when transcript released
            'historical_price': float(current_price),  # The price from the data (quarter end)
            'predicted_price': float(predicted_price),
            'price_change': float(predicted_price - live_current_price),
            'price_change_pct': float((predicted_price - live_current_price) / live_current_price * 100),
            'delta_log': delta_log.item(),
            'input_sequence_ends': input_sequence_end_date.strftime('%Y-%m-%d'),
            'prediction_for_quarter': prediction_target_date.strftime('%Y-%m-%d'),
            'is_future_prediction': is_future_prediction,
            'attn_weights': attn_weights[0].cpu().numpy() if attn_weights is not None else None,
            'quarters_used': quarter_dates,
            'features_available': features_available,
            'features_total': len(feature_cols)
        }

        # If prediction target date is in the past, fetch actual price for comparison
        if not is_future_prediction:
            logger.info(f"Prediction target date {prediction_target_date.strftime('%Y-%m-%d')} is in the past, fetching actual price for comparison")
            actual_price = self.get_actual_price_for_date(ticker, prediction_target_date)
            if actual_price is not None:
                result['actual_price'] = float(actual_price)
                result['actual_price_date'] = prediction_target_date.strftime('%Y-%m-%d')
                result['actual_change_from_transcript'] = float(actual_price - current_price)
                result['actual_change_from_transcript_pct'] = float((actual_price - current_price) / current_price * 100)
                result['prediction_error'] = float(abs(predicted_price - actual_price))
                result['prediction_error_pct'] = float(abs(predicted_price - actual_price) / actual_price * 100)
                result['direction_correct'] = (predicted_price > current_price) == (actual_price > current_price)

        return result

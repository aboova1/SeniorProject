#!/usr/bin/env python3
"""
Live Stock Price Predictor using FMP API
Fetches recent data for a ticker and makes predictions for the current quarter
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LivePredictor:
    """Fetch live data using FMP API and make predictions for current quarter"""

    def __init__(self, api_key: str = "FlVMN0WC4KpV1f9OWTNVamh39gW8wx7L"):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.quarters_needed = 7  # Need 7 quarters to predict Q8

    def _make_request(self, endpoint: str, params: dict = None):
        """Make FMP API request with rate limiting"""
        if params is None:
            params = {}
        params['apikey'] = self.api_key

        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            time.sleep(0.05)  # Rate limiting
            return response.json()
        except Exception as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            return None

    def get_recent_quarters(self):
        """Get the last 7 quarter end dates"""
        today = pd.Timestamp.now()

        # Get last 7 quarter ends
        quarter_ends = []
        for i in range(7, 0, -1):
            quarter_end = (pd.Period(today, freq='Q') - i).end_time
            quarter_ends.append(quarter_end)

        return quarter_ends

    def fetch_historical_prices(self, ticker: str, start_date: str):
        """Fetch historical daily prices from FMP"""
        try:
            logger.info(f"Fetching price data for {ticker} from {start_date}")

            # FMP endpoint for historical prices
            data = self._make_request(f"historical-price-full/{ticker}", {
                'from': start_date,
                'to': datetime.now().strftime('%Y-%m-%d')
            })

            if data and 'historical' in data:
                prices_df = pd.DataFrame(data['historical'])
                prices_df['date'] = pd.to_datetime(prices_df['date'])
                prices_df = prices_df.sort_values('date')
                prices_df.set_index('date', inplace=True)

                logger.info(f"Fetched {len(prices_df)} price records for {ticker}")
                return prices_df

            logger.error(f"No price data returned for {ticker}")
            return None

        except Exception as e:
            logger.error(f"Error fetching prices for {ticker}: {e}")
            return None

    def fetch_quarterly_financials(self, ticker: str):
        """Fetch quarterly financial statements from FMP"""
        try:
            logger.info(f"Fetching quarterly financials for {ticker}")

            # Income statement
            income = self._make_request(f"income-statement/{ticker}", {'period': 'quarter', 'limit': 20})

            # Balance sheet
            balance = self._make_request(f"balance-sheet-statement/{ticker}", {'period': 'quarter', 'limit': 20})

            # Cash flow
            cashflow = self._make_request(f"cash-flow-statement/{ticker}", {'period': 'quarter', 'limit': 20})

            return {
                'income': pd.DataFrame(income) if income else pd.DataFrame(),
                'balance': pd.DataFrame(balance) if balance else pd.DataFrame(),
                'cashflow': pd.DataFrame(cashflow) if cashflow else pd.DataFrame()
            }

        except Exception as e:
            logger.error(f"Error fetching financials for {ticker}: {e}")
            return {'income': pd.DataFrame(), 'balance': pd.DataFrame(), 'cashflow': pd.DataFrame()}

    def build_quarter_features(self, ticker: str, quarter_ends: list):
        """
        Build feature set for the last 7 quarters of a ticker
        Mimics the feature engineering pipeline
        """
        try:
            # Fetch data for the entire period (need extra history for technical indicators)
            start_date = (quarter_ends[0] - timedelta(days=730)).strftime('%Y-%m-%d')

            # Get price data
            prices = self.fetch_historical_prices(ticker, start_date)

            if prices is None or len(prices) == 0:
                logger.error(f"No price data available for {ticker}")
                return None

            # Get financial data
            financials = self.fetch_quarterly_financials(ticker)

            quarters_data = []

            for quarter_end in quarter_ends:
                quarter_features = {
                    'ticker': ticker,
                    'fiscal_quarter_end': quarter_end,
                }

                # Get price at quarter end (within 10 days tolerance)
                quarter_end_ts = pd.Timestamp(quarter_end).tz_localize(None)  # Remove timezone

                # Make sure price index is timezone-naive
                if prices.index.tz is not None:
                    prices_clean = prices.copy()
                    prices_clean.index = prices_clean.index.tz_localize(None)
                else:
                    prices_clean = prices

                # Find prices within 10 days of quarter end
                price_mask = (prices_clean.index >= quarter_end_ts - timedelta(days=10)) & \
                             (prices_clean.index <= quarter_end_ts + timedelta(days=10))
                nearby_prices = prices_clean[price_mask]

                if len(nearby_prices) == 0:
                    logger.warning(f"No price data for {ticker} near {quarter_end}")
                    continue

                # Get closest price to quarter end
                time_diffs = abs(nearby_prices.index - quarter_end_ts)
                closest_idx = time_diffs.argmin()
                current_price = nearby_prices.iloc[closest_idx]['close']
                quarter_features['current_price'] = current_price

                # Get all prices up to this quarter
                quarter_prices = prices_clean[prices_clean.index <= quarter_end_ts].copy()

                # Technical indicators
                if len(quarter_prices) >= 252:
                    # 12-month momentum
                    year_ago_price = quarter_prices['close'].iloc[-252]
                    quarter_features['momentum_12m'] = (current_price / year_ago_price) - 1

                    # Volatility
                    quarter_prices['returns'] = quarter_prices['close'].pct_change()
                    if len(quarter_prices) >= 60:
                        quarter_features['vol_60d'] = quarter_prices['returns'].iloc[-60:].std() * np.sqrt(252)
                    else:
                        quarter_features['vol_60d'] = 0
                else:
                    logger.warning(f"Not enough price history for {ticker} at {quarter_end}")
                    quarter_features['momentum_12m'] = 0
                    quarter_features['vol_60d'] = 0

                # Get fundamental data
                try:
                    if not financials['income'].empty:
                        financials['income']['date'] = pd.to_datetime(financials['income']['date'])

                        # Find closest quarter before quarter_end
                        fin_before = financials['income'][financials['income']['date'] <= quarter_end_ts]

                        if not fin_before.empty:
                            latest_fin = fin_before.iloc[0]  # Most recent

                            quarter_features['revenue'] = latest_fin.get('revenue', 0)
                            quarter_features['operating_income'] = latest_fin.get('operatingIncome', 0)
                            quarter_features['net_income'] = latest_fin.get('netIncome', 0)
                            quarter_features['eps'] = latest_fin.get('eps', 0)

                except Exception as e:
                    logger.debug(f"Could not get financials for {ticker} at {quarter_end}: {e}")

                quarters_data.append(quarter_features)

            if len(quarters_data) < 7:
                logger.error(f"Only found {len(quarters_data)} quarters of data for {ticker}, need 7")
                return None

            logger.info(f"Successfully built {len(quarters_data)} quarters of features for {ticker}")
            return pd.DataFrame(quarters_data)

        except Exception as e:
            logger.error(f"Error building features for {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def prepare_for_model(self, quarters_df: pd.DataFrame, feature_cols: list, scaler):
        """
        Prepare the 7-quarter sequence for model input
        Match the exact feature columns from training
        """
        try:
            # Fill missing features with 0 or forward-fill
            for col in feature_cols:
                if col not in quarters_df.columns:
                    quarters_df[col] = 0

            # Select and order features
            feature_data = quarters_df[feature_cols].copy()

            # Fill NaN values
            feature_data = feature_data.ffill().bfill().fillna(0.0)

            # Scale features
            scaled_features = scaler.transform(feature_data)

            return scaled_features, quarters_df['current_price'].iloc[-1]

        except Exception as e:
            logger.error(f"Error preparing data for model: {e}")
            return None, None

    def predict_next_quarter(
        self,
        ticker: str,
        model,
        feature_cols: list,
        scaler,
        device
    ):
        """
        Make a prediction for the next quarter

        Args:
            ticker: Stock ticker symbol
            model: Trained LSTM model
            feature_cols: List of feature column names
            scaler: Fitted StandardScaler
            device: PyTorch device

        Returns:
            dict with prediction results
        """
        import torch

        logger.info(f"Making live prediction for {ticker}...")

        # Get recent quarter ends
        quarter_ends = self.get_recent_quarters()
        logger.info(f"Using quarters: {[q.strftime('%Y-%m-%d') for q in quarter_ends]}")

        # Build feature set
        quarters_df = self.build_quarter_features(ticker, quarter_ends)

        if quarters_df is None:
            return {
                'success': False,
                'error': f'Could not fetch sufficient data for {ticker}'
            }

        # Prepare for model
        X_scaled, current_price = self.prepare_for_model(quarters_df, feature_cols, scaler)

        if X_scaled is None:
            return {
                'success': False,
                'error': 'Could not prepare features for model'
            }

        # Convert to tensor
        X_tensor = torch.from_numpy(X_scaled.astype(np.float32)).unsqueeze(0).to(device)

        # Get y7_log
        y7_log = np.log1p(current_price)
        y7_log_tensor = torch.tensor([[y7_log]], dtype=torch.float32).to(device)

        # Make prediction
        with torch.no_grad():
            y8_log_hat, delta_log, attn_weights = model(X_tensor, y7_log_tensor)

        # Convert back to price
        predicted_log_price = y8_log_hat.item()
        predicted_price = np.expm1(predicted_log_price)

        # Get next quarter date
        next_quarter = (pd.Period.now(freq='Q') + 1).end_time

        return {
            'success': True,
            'ticker': ticker,
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'price_change': float(predicted_price - current_price),
            'price_change_pct': float((predicted_price - current_price) / current_price * 100),
            'delta_log': delta_log.item(),
            'input_sequence_ends': quarter_ends[-1].strftime('%Y-%m-%d'),
            'prediction_for_quarter': next_quarter.strftime('%Y-%m-%d'),
            'attn_weights': attn_weights[0].cpu().numpy() if attn_weights is not None else None,
            'quarters_used': [q.strftime('%Y-%m-%d') for q in quarter_ends],
            'features_available': len([c for c in feature_cols if c in quarters_df.columns]),
            'features_total': len(feature_cols)
        }

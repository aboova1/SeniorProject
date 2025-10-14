#!/usr/bin/env python3
"""
Live Data Fetcher for Real-Time Predictions
Fetches the most recent 7 quarters of data for any ticker on-demand
"""

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveDataFetcher:
    """Fetch live data for a single ticker for real-time predictions"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('FMP_API_KEY', 'FlVMN0WC4KpV1f9OWTNVamh39gW8wx7L')
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def fetch_quarterly_data(self, ticker: str, quarters: int = 8) -> pd.DataFrame:
        """
        Fetch the most recent quarterly financial data for a ticker

        Args:
            ticker: Stock ticker symbol
            quarters: Number of quarters to fetch

        Returns:
            DataFrame with quarterly data
        """
        try:
            # Fetch income statement data (quarterly)
            url = f"{self.base_url}/income-statement/{ticker}?period=quarter&limit={quarters}&apikey={self.api_key}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            income_data = response.json()

            if not income_data:
                logger.error(f"No income statement data for {ticker}")
                return None

            # Fetch balance sheet data (quarterly)
            url = f"{self.base_url}/balance-sheet-statement/{ticker}?period=quarter&limit={quarters}&apikey={self.api_key}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            balance_data = response.json()

            # Fetch cash flow data (quarterly)
            url = f"{self.base_url}/cash-flow-statement/{ticker}?period=quarter&limit={quarters}&apikey={self.api_key}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            cashflow_data = response.json()

            # Fetch key metrics (quarterly)
            url = f"{self.base_url}/key-metrics/{ticker}?period=quarter&limit={quarters}&apikey={self.api_key}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            metrics_data = response.json()

            # Combine all data
            quarters_list = []

            for i, inc in enumerate(income_data[:quarters]):
                quarter_date = pd.to_datetime(inc['date'])

                # Find matching balance sheet
                bal = next((b for b in balance_data if b['date'] == inc['date']), {})
                cf = next((c for c in cashflow_data if c['date'] == inc['date']), {})
                met = next((m for m in metrics_data if m['date'] == inc['date']), {})

                quarter_data = {
                    'ticker': ticker,
                    'fiscal_quarter_end': quarter_date,
                    'date': inc['date'],
                    'period': inc.get('period', 'Q?'),
                    'calendarYear': inc.get('calendarYear'),

                    # Income statement
                    'revenue': inc.get('revenue'),
                    'costOfRevenue': inc.get('costOfRevenue'),
                    'grossProfit': inc.get('grossProfit'),
                    'operatingIncome': inc.get('operatingIncome'),
                    'netIncome': inc.get('netIncome'),
                    'eps': inc.get('eps'),
                    'epsdiluted': inc.get('epsdiluted'),

                    # Balance sheet
                    'totalAssets': bal.get('totalAssets'),
                    'totalLiabilities': bal.get('totalLiabilities'),
                    'totalEquity': bal.get('totalStockholdersEquity'),
                    'cashAndCashEquivalents': bal.get('cashAndCashEquivalents'),
                    'totalDebt': bal.get('totalDebt'),

                    # Cash flow
                    'operatingCashFlow': cf.get('operatingCashFlow'),
                    'capitalExpenditure': cf.get('capitalExpenditure'),
                    'freeCashFlow': cf.get('freeCashFlow'),

                    # Key metrics
                    'peRatio': met.get('peRatio'),
                    'priceToSalesRatio': met.get('priceToSalesRatio'),
                    'marketCap': met.get('marketCap'),
                    'enterpriseValue': met.get('enterpriseValue'),
                    'evToSales': met.get('evToSales'),
                    'evToOperatingCashFlow': met.get('evToOperatingCashFlow'),
                    'dividendYield': met.get('dividendYield'),
                    'payoutRatio': met.get('payoutRatio'),
                }

                quarters_list.append(quarter_data)

            df = pd.DataFrame(quarters_list)
            df = df.sort_values('fiscal_quarter_end').reset_index(drop=True)

            logger.info(f"Fetched {len(df)} quarters for {ticker} from {df['fiscal_quarter_end'].min()} to {df['fiscal_quarter_end'].max()}")
            return df

        except Exception as e:
            logger.error(f"Error fetching quarterly data for {ticker}: {e}")
            return None

    def fetch_historical_prices(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Fetch historical daily prices for a ticker"""
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            url = f"{self.base_url}/historical-price-full/{ticker}"
            params = {
                'from': start_date,
                'to': end_date,
                'apikey': self.api_key
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'historical' not in data or not data['historical']:
                logger.warning(f"No price data for {ticker}")
                return pd.DataFrame()

            prices_df = pd.DataFrame(data['historical'])
            prices_df['date'] = pd.to_datetime(prices_df['date'])
            prices_df = prices_df.sort_values('date')

            return prices_df

        except Exception as e:
            logger.error(f"Error fetching prices for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_earnings_dates(self, ticker: str) -> pd.DataFrame:
        """Fetch earnings call dates (when transcripts were released)"""
        try:
            # Try to get earnings calendar data
            url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{ticker}"
            params = {'apikey': self.api_key}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                return pd.DataFrame()

            earnings_df = pd.DataFrame(data)
            earnings_df['date'] = pd.to_datetime(earnings_df['date'])
            earnings_df['fiscalDateEnding'] = pd.to_datetime(earnings_df['fiscalDateEnding'])

            # Use 'date' as the transcript release date
            earnings_df = earnings_df.rename(columns={
                'date': 'period_start_date',
                'fiscalDateEnding': 'fiscal_quarter_end'
            })

            return earnings_df[['ticker', 'fiscal_quarter_end', 'period_start_date']]

        except Exception as e:
            logger.warning(f"Could not fetch earnings dates for {ticker}: {e}")
            return pd.DataFrame()

    def add_prices_to_quarters(self, quarters_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Add price data to quarters dataframe"""
        if quarters_df is None or len(quarters_df) == 0:
            return None

        # Fetch prices from earliest quarter to now
        start_date = (quarters_df['fiscal_quarter_end'].min() - timedelta(days=10)).strftime('%Y-%m-%d')
        prices_df = self.fetch_historical_prices(ticker, start_date)

        if prices_df.empty:
            logger.error(f"No price data available for {ticker}")
            return None

        # Add current_price for each quarter (price on the quarter end date, or closest available)
        quarters_df = quarters_df.copy()
        quarters_df['current_price'] = np.nan

        for idx, row in quarters_df.iterrows():
            quarter_end = row['fiscal_quarter_end']

            # Find price on or near quarter end
            nearby_prices = prices_df[
                (prices_df['date'] >= quarter_end - timedelta(days=5)) &
                (prices_df['date'] <= quarter_end + timedelta(days=5))
            ]

            if not nearby_prices.empty:
                closest_price = nearby_prices.iloc[(nearby_prices['date'] - quarter_end).abs().argmin()]
                quarters_df.at[idx, 'current_price'] = closest_price['close']
            else:
                logger.warning(f"No price data near {quarter_end} for {ticker}")

        return quarters_df

    def add_earnings_dates_to_quarters(self, quarters_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Add earnings call dates (transcript release dates) to quarters"""
        if quarters_df is None or len(quarters_df) == 0:
            return None

        earnings_df = self.fetch_earnings_dates(ticker)

        if earnings_df.empty:
            logger.warning(f"No earnings dates found for {ticker}, using generic estimates")
            # Estimate: transcripts released ~45 days after quarter end
            quarters_df['period_start_date'] = quarters_df['fiscal_quarter_end'] + timedelta(days=45)
            return quarters_df

        # Merge earnings dates with quarters
        quarters_df = quarters_df.merge(
            earnings_df[['fiscal_quarter_end', 'period_start_date']],
            on='fiscal_quarter_end',
            how='left'
        )

        # For any missing dates, estimate as 45 days after quarter end
        missing_mask = quarters_df['period_start_date'].isna()
        if missing_mask.any():
            logger.warning(f"Estimating {missing_mask.sum()} missing earnings dates for {ticker}")
            quarters_df.loc[missing_mask, 'period_start_date'] = \
                quarters_df.loc[missing_mask, 'fiscal_quarter_end'] + timedelta(days=45)

        return quarters_df

    def calculate_holding_periods(self, quarters_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate holding period (days between consecutive transcript releases)"""
        if quarters_df is None or len(quarters_df) == 0:
            return None

        quarters_df = quarters_df.sort_values('fiscal_quarter_end').copy()

        # period_end_date is when the NEXT quarter's transcript is released
        quarters_df['period_end_date'] = quarters_df['period_start_date'].shift(-1)

        # Calculate holding period in days
        quarters_df['holding_period_days'] = (
            quarters_df['period_end_date'] - quarters_df['period_start_date']
        ).dt.days

        # For the last quarter (no next quarter), estimate based on average
        if pd.notna(quarters_df['holding_period_days'].iloc[:-1].mean()):
            avg_holding = quarters_df['holding_period_days'].iloc[:-1].mean()
            quarters_df.loc[quarters_df.index[-1], 'holding_period_days'] = avg_holding

        return quarters_df

    def fetch_complete_data(self, ticker: str, quarters: int = 7) -> pd.DataFrame:
        """
        Fetch complete data for a ticker ready for model prediction

        Args:
            ticker: Stock ticker symbol
            quarters: Number of quarters needed (default 7 for input to model)

        Returns:
            DataFrame with complete quarterly data including prices and dates
        """
        logger.info(f"Fetching live data for {ticker}...")

        # Fetch quarterly fundamentals (get extra in case some are incomplete)
        quarters_df = self.fetch_quarterly_data(ticker, quarters=quarters + 3)

        if quarters_df is None or len(quarters_df) == 0:
            logger.error(f"No quarterly data available for {ticker}")
            return None

        # Add prices
        quarters_df = self.add_prices_to_quarters(quarters_df, ticker)

        if quarters_df is None:
            return None

        # Add earnings dates
        quarters_df = self.add_earnings_dates_to_quarters(quarters_df, ticker)

        # Calculate holding periods
        quarters_df = self.calculate_holding_periods(quarters_df)

        # Filter to only complete quarters (has price data)
        quarters_df = quarters_df[quarters_df['current_price'].notna()].copy()

        # Take most recent N quarters
        quarters_df = quarters_df.tail(quarters)

        # Add in_sp500 flag (for compatibility)
        quarters_df['in_sp500'] = True

        logger.info(f"Successfully fetched {len(quarters_df)} complete quarters for {ticker}")
        logger.info(f"Date range: {quarters_df['fiscal_quarter_end'].min()} to {quarters_df['fiscal_quarter_end'].max()}")

        return quarters_df


# Test function
if __name__ == "__main__":
    fetcher = LiveDataFetcher()

    # Test with AAPL
    data = fetcher.fetch_complete_data('AAPL', quarters=7)

    if data is not None:
        print("\n=== AAPL Live Data ===")
        print(data[['ticker', 'fiscal_quarter_end', 'period_start_date', 'current_price', 'revenue', 'netIncome']])
        print(f"\nShape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")

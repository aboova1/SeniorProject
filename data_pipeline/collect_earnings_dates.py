#!/usr/bin/env python3
"""
Earnings Date Collector - Fetch Actual Earnings Dates from FMP
================================================================

Collects actual earnings announcement dates and fiscal periods from
Financial Modeling Prep API to replace inferred calendar quarters.

This provides accurate fiscal quarter matching for all companies,
regardless of their fiscal year calendar.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EarningsDateCollector:
    """Collect actual earnings dates from FMP API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url_v3 = "https://financialmodelingprep.com/api/v3"
        self.base_url_v4 = "https://financialmodelingprep.com/api/v4"
        self.session = requests.Session()
        self.data_dir = Path("data_pipeline/data")
        self.data_dir.mkdir(exist_ok=True, parents=True)

    def _make_request(self, endpoint: str, base_url: str = None, params: Dict = None) -> Dict:
        """Make API request with rate limiting"""
        if params is None:
            params = {}
        params['apikey'] = self.api_key

        if base_url is None:
            base_url = self.base_url_v3

        url = f"{base_url}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(0.1)  # Rate limiting
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 403:
                logger.error(f"API access forbidden for {endpoint}. This endpoint may require a paid plan.")
            else:
                logger.error(f"HTTP error for {endpoint}: {e}")
            return []
        except Exception as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            return []

    def collect_earnings_calendar_range(self, start_date: str = '2005-01-01',
                                       end_date: str = '2025-12-31') -> pd.DataFrame:
        """
        Collect earnings dates for all companies in date range using earnings calendar.

        This is the most comprehensive method as it gets all earnings in one call.
        """
        logger.info(f"Collecting earnings calendar from {start_date} to {end_date}...")

        # Split into yearly chunks to avoid overwhelming the API
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        all_earnings = []

        # Process year by year
        for year in range(start.year, end.year + 1):
            year_start = max(start, pd.Timestamp(f'{year}-01-01')).strftime('%Y-%m-%d')
            year_end = min(end, pd.Timestamp(f'{year}-12-31')).strftime('%Y-%m-%d')

            logger.info(f"  Fetching {year}...")

            params = {
                'from': year_start,
                'to': year_end
            }

            data = self._make_request('earning_calendar', params=params)

            if data and isinstance(data, list):
                all_earnings.extend(data)
                logger.info(f"    Found {len(data)} earnings announcements")
            else:
                logger.warning(f"    No data returned for {year}")

            time.sleep(0.2)  # Extra delay between years

        if not all_earnings:
            logger.warning("No earnings data collected!")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_earnings)

        # Rename columns to match our schema
        column_mapping = {
            'symbol': 'ticker',
            'date': 'earnings_date',
            'fiscalDateEnding': 'fiscal_quarter_end',
            'time': 'earnings_time'
        }

        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Ensure datetime types
        df['earnings_date'] = pd.to_datetime(df['earnings_date'])
        if 'fiscal_quarter_end' in df.columns:
            df['fiscal_quarter_end'] = pd.to_datetime(df['fiscal_quarter_end'])

        logger.info(f"Collected {len(df)} total earnings announcements for {df['ticker'].nunique()} companies")

        return df

    def collect_historical_earnings_by_ticker(self, tickers: List[str]) -> pd.DataFrame:
        """
        Collect historical earnings for specific tickers.

        Fallback method if earnings calendar doesn't work.
        """
        logger.info(f"Collecting historical earnings for {len(tickers)} tickers...")

        all_earnings = []
        failed_tickers = []

        for i, ticker in enumerate(tickers):
            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i+1}/{len(tickers)}")

            endpoint = f"historical/earning_calendar/{ticker}"
            data = self._make_request(endpoint)

            if data and isinstance(data, list):
                for earning in data:
                    earning['ticker'] = ticker
                    all_earnings.append(earning)
            else:
                failed_tickers.append(ticker)

            time.sleep(0.1)

        if failed_tickers:
            logger.warning(f"Failed to fetch data for {len(failed_tickers)} tickers: {failed_tickers[:10]}...")

        if not all_earnings:
            logger.warning("No earnings data collected!")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_earnings)

        # Rename columns
        column_mapping = {
            'date': 'earnings_date',
            'fiscalDateEnding': 'fiscal_quarter_end',
            'time': 'earnings_time'
        }

        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Ensure datetime types
        df['earnings_date'] = pd.to_datetime(df['earnings_date'])
        if 'fiscal_quarter_end' in df.columns:
            df['fiscal_quarter_end'] = pd.to_datetime(df['fiscal_quarter_end'])

        logger.info(f"Collected {len(df)} earnings announcements for {df['ticker'].nunique()} companies")

        return df

    def collect_and_save(self, method: str = 'calendar',
                         start_date: str = '2005-01-01',
                         end_date: str = '2025-12-31',
                         tickers: List[str] = None) -> pd.DataFrame:
        """
        Main method to collect and save earnings dates.

        Args:
            method: 'calendar' (date range) or 'ticker' (per-ticker historical)
            start_date: Start date for calendar method
            end_date: End date for calendar method
            tickers: List of tickers for ticker method

        Returns:
            DataFrame with earnings dates
        """
        logger.info("="*70)
        logger.info("EARNINGS DATE COLLECTION")
        logger.info("="*70)

        if method == 'calendar':
            df = self.collect_earnings_calendar_range(start_date, end_date)
        elif method == 'ticker':
            if tickers is None:
                raise ValueError("tickers list required for 'ticker' method")
            df = self.collect_historical_earnings_by_ticker(tickers)
        else:
            raise ValueError(f"Unknown method: {method}")

        if df.empty:
            logger.error("No data collected!")
            return df

        # Save to parquet
        output_path = self.data_dir / "earnings_dates.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"\nSaved earnings dates to {output_path}")

        # Print summary statistics
        self._print_summary(df)

        return df

    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics of earnings data"""
        logger.info("\n" + "="*70)
        logger.info("EARNINGS DATA SUMMARY")
        logger.info("="*70)
        logger.info(f"Total earnings announcements: {len(df):,}")
        logger.info(f"Unique companies: {df['ticker'].nunique()}")
        logger.info(f"Date range: {df['earnings_date'].min()} to {df['earnings_date'].max()}")

        if 'fiscal_quarter_end' in df.columns:
            # Calculate average gap between fiscal quarter end and earnings date
            df_with_fiscal = df.dropna(subset=['fiscal_quarter_end'])
            if len(df_with_fiscal) > 0:
                df_with_fiscal['days_after_quarter'] = (
                    df_with_fiscal['earnings_date'] - df_with_fiscal['fiscal_quarter_end']
                ).dt.days

                avg_gap = df_with_fiscal['days_after_quarter'].mean()
                median_gap = df_with_fiscal['days_after_quarter'].median()

                logger.info(f"\nTiming statistics:")
                logger.info(f"  Average gap: {avg_gap:.1f} days after fiscal quarter end")
                logger.info(f"  Median gap: {median_gap:.1f} days")

        if 'earnings_time' in df.columns:
            time_dist = df['earnings_time'].value_counts()
            logger.info(f"\nEarnings timing:")
            for time_type, count in time_dist.items():
                logger.info(f"  {time_type}: {count} ({count/len(df)*100:.1f}%)")

        logger.info("="*70)


def main():
    """Example usage"""
    import os

    # Get API key from environment
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        logger.error("FMP_API_KEY environment variable not set!")
        logger.info("Set it with: export FMP_API_KEY='your_key_here'")
        return

    collector = EarningsDateCollector(api_key)

    # Try calendar method first (most efficient)
    logger.info("Attempting to collect earnings calendar...")
    df = collector.collect_and_save(
        method='calendar',
        start_date='2005-01-01',
        end_date='2025-12-31'
    )

    if df.empty:
        logger.warning("\nCalendar method failed. Trying ticker-by-ticker method...")
        logger.warning("This requires a list of tickers. Loading from fundamentals...")

        # Load tickers from existing data
        fundamentals = pd.read_parquet('data_pipeline/data/fundamentals_quarterly.parquet')
        tickers = sorted(fundamentals['ticker'].unique())

        logger.info(f"Found {len(tickers)} tickers in fundamentals data")

        df = collector.collect_and_save(
            method='ticker',
            tickers=tickers
        )


if __name__ == "__main__":
    main()

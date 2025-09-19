#!/usr/bin/env python3
"""
Precise S&P 500 Historical Data Collector
Uses FMP's historical constituent changes to build exact quarterly membership
Target: ~40,000 quarterly observations from 2005-2025
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
from typing import List, Dict, Set
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreciseSP500Collector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.session = requests.Session()
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting"""
        if params is None:
            params = {}
        params['apikey'] = self.api_key

        url = f"{self.base_url}/{endpoint}"

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            time.sleep(0.05)  # Rate limiting
            return response.json()
        except Exception as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            return {}

    def get_historical_sp500_changes(self) -> pd.DataFrame:
        """Get all historical S&P 500 constituent changes"""
        logger.info("Downloading complete S&P 500 historical changes...")

        changes_data = self._make_request("historical/sp500_constituent")

        if changes_data:
            changes_df = pd.DataFrame(changes_data)
            changes_df['date'] = pd.to_datetime(changes_df['date'])

            # Sort by date (oldest first)
            changes_df = changes_df.sort_values('date')

            logger.info(f"Downloaded {len(changes_df)} historical S&P 500 changes")
            return changes_df

        return pd.DataFrame()

    def build_quarterly_membership(self, changes_df: pd.DataFrame) -> pd.DataFrame:
        """Build precise quarterly S&P 500 membership from 2005-2025"""
        logger.info("Building precise quarterly S&P 500 membership...")

        # Generate all quarter ends from 2005 to 2025
        quarter_ends = pd.date_range(start='2005-03-31', end='2025-12-31', freq='Q')

        # Get current S&P 500 as baseline
        current_sp500 = self._make_request("sp500_constituent")
        if not current_sp500:
            logger.error("Failed to get current S&P 500")
            return pd.DataFrame()

        current_tickers = set(item['symbol'] for item in current_sp500)

        # Build membership for each quarter by working backwards from current
        quarterly_membership = []

        for quarter_end in reversed(quarter_ends):
            # Start with current membership
            active_tickers = current_tickers.copy()

            # Apply all changes that happened after this quarter
            future_changes = changes_df[changes_df['date'] > quarter_end]

            for _, change in future_changes.iterrows():
                # Reverse the change: if a ticker was added later, remove it from this quarter
                if change['symbol'] in active_tickers:
                    active_tickers.remove(change['symbol'])

                # If a ticker was removed later, add it back to this quarter
                if 'removedTicker' in change and pd.notna(change['removedTicker']):
                    active_tickers.add(change['removedTicker'])

            # Record membership for this quarter
            for ticker in active_tickers:
                quarterly_membership.append({
                    'quarter_end': quarter_end,
                    'ticker': ticker,
                    'in_sp500': True
                })

        membership_df = pd.DataFrame(quarterly_membership)
        membership_df = membership_df.sort_values(['quarter_end', 'ticker'])

        logger.info(f"Built quarterly membership: {len(membership_df)} ticker-quarter observations")
        logger.info(f"Quarters: {len(quarter_ends)}, Average tickers per quarter: {len(membership_df)/len(quarter_ends):.0f}")

        # Save the membership data
        membership_df.to_parquet(self.data_dir / "sp500_quarterly_membership.parquet", index=False)

        return membership_df

    def download_comprehensive_data(self, membership_df: pd.DataFrame):
        """Download all data for precise S&P 500 universe"""

        # Get unique tickers (filter out empty/NaN values)
        all_tickers = sorted([t for t in membership_df['ticker'].unique() if t and pd.notna(t)])
        logger.info(f"Downloading data for {len(all_tickers)} unique S&P 500 companies")

        # Download price data
        self.download_price_data(all_tickers)

        # Download fundamental data
        self.download_fundamental_data(all_tickers)

        # Download earnings data
        self.download_earnings_data(all_tickers)

        # Download text data
        self.download_text_data(all_tickers)

    def download_price_data(self, tickers: List[str]):
        """Download daily price data from 2005"""
        logger.info(f"Downloading price data for {len(tickers)} tickers...")

        all_prices = []

        for i, ticker in enumerate(tickers):
            if not ticker or pd.isna(ticker):  # Skip empty/NaN tickers
                continue

            if i % 25 == 0:
                logger.info(f"Price progress: {i+1}/{len(tickers)} ({ticker})")

            data = self._make_request(f"historical-price-full/{ticker}", {
                'from': '2005-01-01',
                'to': datetime.now().strftime('%Y-%m-%d')
            })

            if data and 'historical' in data:
                price_data = pd.DataFrame(data['historical'])
                price_data['ticker'] = ticker
                price_data['date'] = pd.to_datetime(price_data['date'])
                all_prices.append(price_data[['ticker', 'date', 'adjClose', 'volume']])

        if all_prices:
            prices_df = pd.concat(all_prices, ignore_index=True)
            prices_df = prices_df.rename(columns={'adjClose': 'adj_close'})
            prices_df.to_parquet(self.data_dir / "prices_daily.parquet", index=False)
            logger.info(f"Saved {len(prices_df)} daily price observations")

    def download_fundamental_data(self, tickers: List[str]):
        """Download quarterly fundamentals from 2005"""
        logger.info(f"Downloading fundamentals for {len(tickers)} tickers...")

        all_fundamentals = []

        for i, ticker in enumerate(tickers):
            if not ticker or pd.isna(ticker):  # Skip empty/NaN tickers
                continue

            if i % 25 == 0:
                logger.info(f"Fundamentals progress: {i+1}/{len(tickers)} ({ticker})")

            # Get 80 quarters (20 years) of data
            income_data = self._make_request(f"income-statement/{ticker}", {'period': 'quarter', 'limit': 80})
            balance_data = self._make_request(f"balance-sheet-statement/{ticker}", {'period': 'quarter', 'limit': 80})
            cashflow_data = self._make_request(f"cash-flow-statement/{ticker}", {'period': 'quarter', 'limit': 80})
            metrics_data = self._make_request(f"key-metrics/{ticker}", {'period': 'quarter', 'limit': 80})
            ratios_data = self._make_request(f"ratios/{ticker}", {'period': 'quarter', 'limit': 80})

            # Combine all data sources
            max_len = max(len(income_data or []), len(balance_data or []), len(cashflow_data or []))

            for j in range(max_len):
                row = {'ticker': ticker}

                # Income statement
                if income_data and j < len(income_data):
                    inc = income_data[j]
                    row.update({
                        'fiscal_quarter_end': inc.get('date'),
                        'revenue': inc.get('revenue'),
                        'net_income': inc.get('netIncome'),
                        'eps': inc.get('eps'),
                        'operating_income': inc.get('operatingIncome'),
                        'gross_profit': inc.get('grossProfit'),
                        'ebitda': inc.get('ebitda'),
                    })

                # Balance sheet
                if balance_data and j < len(balance_data):
                    bal = balance_data[j]
                    row.update({
                        'total_assets': bal.get('totalAssets'),
                        'total_debt': bal.get('totalDebt'),
                        'stockholders_equity': bal.get('totalStockholdersEquity'),
                        'current_assets': bal.get('totalCurrentAssets'),
                        'current_liabilities': bal.get('totalCurrentLiabilities'),
                    })

                # Cash flow
                if cashflow_data and j < len(cashflow_data):
                    cf = cashflow_data[j]
                    row.update({
                        'free_cash_flow': cf.get('freeCashFlow'),
                        'operating_cash_flow': cf.get('operatingCashFlow'),
                        'capital_expenditure': cf.get('capitalExpenditure'),
                    })

                # Metrics
                if metrics_data and j < len(metrics_data):
                    met = metrics_data[j]
                    row.update({
                        'pe_ratio': met.get('peRatio'),
                        'market_cap': met.get('marketCap'),
                        'enterprise_value': met.get('enterpriseValue'),
                    })

                # Ratios
                if ratios_data and j < len(ratios_data):
                    rat = ratios_data[j]
                    row.update({
                        'current_ratio': rat.get('currentRatio'),
                        'debt_to_equity': rat.get('debtEquityRatio'),
                        'roe': rat.get('returnOnEquity'),
                        'operating_margin': rat.get('operatingProfitMargin'),
                    })

                if row.get('fiscal_quarter_end'):
                    all_fundamentals.append(row)

        if all_fundamentals:
            fund_df = pd.DataFrame(all_fundamentals)
            fund_df['fiscal_quarter_end'] = pd.to_datetime(fund_df['fiscal_quarter_end'])
            fund_df = fund_df[fund_df['fiscal_quarter_end'] >= '2005-01-01']
            fund_df.to_parquet(self.data_dir / "fundamentals_quarterly.parquet", index=False)
            logger.info(f"Saved {len(fund_df)} quarterly fundamental observations")

    def download_earnings_data(self, tickers: List[str]):
        """Download earnings surprise data"""
        logger.info(f"Downloading earnings for {len(tickers)} tickers...")

        all_earnings = []

        for i, ticker in enumerate(tickers):
            if not ticker or pd.isna(ticker):  # Skip empty/NaN tickers
                continue

            if i % 25 == 0:
                logger.info(f"Earnings progress: {i+1}/{len(tickers)} ({ticker})")

            earnings_data = self._make_request(f"earnings-surprises/{ticker}")

            if earnings_data:
                for earning in earnings_data:
                    date_str = earning.get('date')
                    if date_str and pd.to_datetime(date_str) >= pd.to_datetime('2005-01-01'):
                        all_earnings.append({
                            'ticker': ticker,
                            'date': date_str,
                            'actual_eps': earning.get('actualEarningResult'),
                            'estimated_eps': earning.get('estimatedEarning'),
                        })

        if all_earnings:
            earn_df = pd.DataFrame(all_earnings)
            earn_df['date'] = pd.to_datetime(earn_df['date'])
            earn_df['earnings_surprise_pct'] = (
                (earn_df['actual_eps'] - earn_df['estimated_eps']) /
                earn_df['estimated_eps'].abs()
            ) * 100
            earn_df.to_parquet(self.data_dir / "earnings_data.parquet", index=False)
            logger.info(f"Saved {len(earn_df)} earnings observations")

    def download_text_data(self, tickers: List[str]):
        """Download earnings call transcripts only - comprehensive historical coverage"""
        logger.info(f"Downloading earnings call transcripts for {len(tickers)} tickers...")

        all_text = []

        for i, ticker in enumerate(tickers):
            if not ticker or pd.isna(ticker):  # Skip empty/NaN tickers
                continue

            if i % 25 == 0:
                logger.info(f"Text progress: {i+1}/{len(tickers)} ({ticker})")

            # Earnings transcripts - get all available historical data
            transcripts = self._make_request(f"earning_call_transcript/{ticker}")
            if transcripts and isinstance(transcripts, list) and len(transcripts) > 0:
                for transcript in transcripts:  # Get ALL transcripts, not just recent ones
                    date_str = transcript.get('date')
                    content = transcript.get('content', '')
                    if date_str and content:
                        # Filter to reasonable date range (2005-2025)
                        try:
                            filing_date = pd.to_datetime(date_str)
                            if filing_date >= pd.to_datetime('2005-01-01') and filing_date <= pd.to_datetime('2025-12-31'):
                                # Keep full transcript content - no truncation for better text analysis
                                all_text.append({
                                    'ticker': ticker,
                                    'filing_date': date_str,
                                    'document_type': 'earnings_call',
                                    'text_content': content,
                                    'quarter': transcript.get('quarter'),
                                    'year': transcript.get('year'),
                                    'url': None  # Transcripts don't have URLs
                                })
                        except:
                            # Skip records with invalid dates
                            continue

        if all_text:
            text_df = pd.DataFrame(all_text)
            text_df['filing_date'] = pd.to_datetime(text_df['filing_date'])
            text_df.to_parquet(self.data_dir / "text_raw.parquet", index=False)
            logger.info(f"Saved {len(text_df)} earnings call transcripts")
        else:
            logger.warning("No text data collected - creating empty file")
            # Create empty file with correct structure
            empty_df = pd.DataFrame(columns=['ticker', 'filing_date', 'document_type', 'text_content', 'quarter', 'year', 'url'])
            empty_df.to_parquet(self.data_dir / "text_raw.parquet", index=False)

def main():
    """Execute precise S&P 500 data collection"""
    api_key = "FlVMN0WC4KpV1f9OWTNVamh39gW8wx7L"
    collector = PreciseSP500Collector(api_key)

    # Step 1: Get historical S&P 500 changes
    changes_df = collector.get_historical_sp500_changes()

    # Step 2: Build precise quarterly membership
    membership_df = collector.build_quarterly_membership(changes_df)

    # Step 3: Download comprehensive data
    collector.download_comprehensive_data(membership_df)

    # Summary
    quarters = membership_df['quarter_end'].nunique()
    companies = membership_df['ticker'].nunique()
    total_obs = len(membership_df)

    logger.info("="*60)
    logger.info("DATA COLLECTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Quarters covered: {quarters} (2005-2025)")
    logger.info(f"Unique companies: {companies}")
    logger.info(f"Total ticker-quarter observations: {total_obs:,}")
    logger.info(f"Target ~40,000 observations: {total_obs/40000*100:.1f}% achieved")
    logger.info("="*60)

if __name__ == "__main__":
    main()
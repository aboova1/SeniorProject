#!/usr/bin/env python3
"""
Enhanced Earnings Call Transcript Collector
Uses v4 API for metadata + v3 API with quarter/year parameters for content
Targets comprehensive historical coverage (2005-2025)
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTranscriptCollector:
    def __init__(self, api_key: str, max_workers: int = 10):
        self.api_key = api_key
        self.base_url_v3 = "https://financialmodelingprep.com/api/v3"
        self.base_url_v4 = "https://financialmodelingprep.com/api/v4"
        self.max_workers = max_workers

        # Configure session with connection pooling and retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=max_workers,
            pool_maxsize=max_workers * 2
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.data_dir = Path("data_pipeline/data")
        self.data_dir.mkdir(exist_ok=True, parents=True)

    def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Make API request with minimal rate limiting"""
        if params is None:
            params = {}
        params['apikey'] = self.api_key

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed for {url}: {e}")
            return {}

    def get_transcript_metadata(self, ticker: str) -> List[Dict]:
        """Get all available quarters for a ticker using v4 API"""
        url = f"{self.base_url_v4}/earning_call_transcript"
        metadata = self._make_request(url, {'symbol': ticker})

        if isinstance(metadata, list):
            # Convert metadata arrays to structured data
            quarters = []
            for meta in metadata:
                if isinstance(meta, list) and len(meta) >= 3:
                    quarter, year, date = meta[0], meta[1], meta[2]
                    quarters.append({
                        'ticker': ticker,
                        'quarter': quarter,
                        'year': year,
                        'date': date
                    })
            return quarters
        return []

    def get_transcript_content(self, ticker: str, quarter: int, year: int) -> Dict:
        """Get actual transcript content using v3 API with quarter/year"""
        url = f"{self.base_url_v3}/earning_call_transcript/{ticker}"
        data = self._make_request(url, {'quarter': quarter, 'year': year})

        if isinstance(data, list) and len(data) > 0:
            transcript = data[0]
            if isinstance(transcript, dict) and transcript.get('content'):
                return transcript
        return {}

    def process_single_ticker(self, ticker: str, membership_df: pd.DataFrame) -> tuple:
        """Process a single ticker - used for parallel processing"""
        if not ticker or pd.isna(ticker):
            return [], ticker

        try:
            # Step 1: Get all available quarters for this ticker
            quarters_metadata = self.get_transcript_metadata(ticker)

            if not quarters_metadata:
                logger.debug(f"{ticker}: No metadata found")
                return [], ticker

            logger.info(f"{ticker}: Found {len(quarters_metadata)} quarters available")

            # Step 2: Fetch actual transcript content for each quarter
            transcripts = []
            for quarter_info in quarters_metadata:
                quarter = quarter_info['quarter']
                year = quarter_info['year']
                date = quarter_info['date']

                # Get actual transcript content
                transcript_data = self.get_transcript_content(ticker, quarter, year)

                if transcript_data and transcript_data.get('content'):
                    try:
                        filing_date = pd.to_datetime(date)
                        if filing_date >= pd.to_datetime('2005-01-01') and filing_date <= pd.to_datetime('2025-12-31'):
                            # Check if ticker was in S&P 500 around this time
                            # Use a more flexible date range (within 45 days of quarter end)
                            quarter_end = pd.Period(filing_date, freq='Q').to_timestamp(how='end')
                            date_lower = quarter_end - pd.Timedelta(days=45)
                            date_upper = quarter_end + pd.Timedelta(days=45)

                            was_in_sp500 = membership_df[
                                (membership_df['ticker'] == ticker) &
                                (membership_df['quarter_end'] >= date_lower) &
                                (membership_df['quarter_end'] <= date_upper) &
                                (membership_df['in_sp500'] == True)
                            ].shape[0] > 0

                            if was_in_sp500:
                                transcripts.append({
                                    'ticker': ticker,
                                    'filing_date': date,
                                    'document_type': 'earnings_call',
                                    'text_content': transcript_data['content'],
                                    'quarter': quarter,
                                    'year': year,
                                    'url': None
                                })
                    except Exception as e:
                        logger.debug(f"Error processing {ticker} Q{quarter} {year}: {e}")
                        continue

            if len(transcripts) == 0:
                return [], ticker
            else:
                return transcripts, None

        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
            return [], ticker

    def download_comprehensive_transcripts(self, tickers: List[str], membership_df: pd.DataFrame):
        """Download all available transcripts for all tickers in parallel, filtered by S&P 500 membership"""
        logger.info(f"Starting parallel transcript collection for {len(tickers)} tickers...")
        logger.info(f"Using {self.max_workers} parallel workers")

        all_transcripts = []
        failed_tickers = []

        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.process_single_ticker, ticker, membership_df): ticker
                for ticker in tickers if ticker and not pd.isna(ticker)
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                completed += 1

                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(future_to_ticker)} tickers completed")
                    logger.info(f"Collected so far: {len(all_transcripts)} transcripts")

                try:
                    transcripts, failed_ticker = future.result()
                    if transcripts:
                        all_transcripts.extend(transcripts)
                        logger.info(f"{ticker}: Collected {len(transcripts)} transcripts")
                    if failed_ticker:
                        failed_tickers.append(failed_ticker)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    failed_tickers.append(ticker)

        # Save results with safety check
        if all_transcripts:
            text_df = pd.DataFrame(all_transcripts)
            text_df['filing_date'] = pd.to_datetime(text_df['filing_date'])

            output_file = self.data_dir / "text_raw_comprehensive.parquet"

            # Safety check: warn if overwriting with significantly less data
            if output_file.exists():
                existing_df = pd.read_parquet(output_file)
                if len(text_df) < len(existing_df) * 0.8:
                    logger.warning("="*60)
                    logger.warning(f"WARNING: New transcript count ({len(text_df)}) is <80% of existing ({len(existing_df)})")
                    logger.warning("This may indicate incomplete collection. Existing file will be backed up.")
                    logger.warning("="*60)
                    backup_file = self.data_dir / f"text_raw_comprehensive_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                    existing_df.to_parquet(backup_file, index=False)
                    logger.info(f"Backup saved to: {backup_file}")

            text_df.to_parquet(output_file, index=False)

            logger.info("="*60)
            logger.info("COMPREHENSIVE TRANSCRIPT COLLECTION COMPLETE")
            logger.info("="*60)
            logger.info(f"Total transcripts collected: {len(text_df):,}")
            logger.info(f"Unique tickers with transcripts: {text_df['ticker'].nunique()}")
            logger.info(f"Date range: {text_df['filing_date'].min()} to {text_df['filing_date'].max()}")
            logger.info(f"Average transcripts per ticker: {len(text_df) / text_df['ticker'].nunique():.1f}")

            # Show distribution by year
            by_year = text_df.groupby(text_df['filing_date'].dt.year).size().sort_index()
            logger.info("\\nTranscripts by year:")
            for year, count in by_year.items():
                logger.info(f"  {year}: {count:,}")

            # Show failed tickers
            if failed_tickers:
                logger.warning(f"\\nFailed to get transcripts for {len(failed_tickers)} tickers:")
                logger.warning(f"  {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")

            logger.info("="*60)

        else:
            logger.error("No transcripts collected!")

def main():
    """Run comprehensive transcript collection"""
    api_key = "FlVMN0WC4KpV1f9OWTNVamh39gW8wx7L"
    collector = EnhancedTranscriptCollector(api_key)

    # Load existing membership data to get tickers
    membership_df = pd.read_parquet("data_pipeline/data/sp500_quarterly_membership.parquet")
    all_tickers = sorted([t for t in membership_df['ticker'].unique() if t and pd.notna(t)])

    logger.info(f"Running ENHANCED transcript collection for {len(all_tickers)} S&P 500 companies")
    logger.info("Strategy: v4 API for metadata + v3 API with quarter/year for content")
    logger.info("Goal: Get ALL available transcripts from quarters when company was in S&P 500")

    # Run enhanced collection
    collector.download_comprehensive_transcripts(all_tickers, membership_df)

if __name__ == "__main__":
    main()
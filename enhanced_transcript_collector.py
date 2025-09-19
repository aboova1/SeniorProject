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
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTranscriptCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url_v3 = "https://financialmodelingprep.com/api/v3"
        self.base_url_v4 = "https://financialmodelingprep.com/api/v4"
        self.session = requests.Session()
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting"""
        if params is None:
            params = {}
        params['apikey'] = self.api_key

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            time.sleep(0.05)  # Rate limiting
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

    def download_comprehensive_transcripts(self, tickers: List[str]):
        """Download all available transcripts for all tickers"""
        logger.info(f"Starting comprehensive transcript collection for {len(tickers)} tickers...")

        all_transcripts = []
        total_transcripts = 0
        failed_tickers = []

        for i, ticker in enumerate(tickers):
            if not ticker or pd.isna(ticker):
                continue

            if i % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(tickers)} tickers ({ticker})")
                logger.info(f"Collected so far: {total_transcripts} transcripts")

            try:
                # Step 1: Get all available quarters for this ticker
                quarters_metadata = self.get_transcript_metadata(ticker)

                if not quarters_metadata:
                    logger.warning(f"No transcript metadata for {ticker}")
                    failed_tickers.append(ticker)
                    continue

                logger.info(f"{ticker}: Found {len(quarters_metadata)} quarters")

                # Step 2: Fetch actual transcript content for each quarter
                ticker_transcripts = 0
                for quarter_info in quarters_metadata:
                    quarter = quarter_info['quarter']
                    year = quarter_info['year']
                    date = quarter_info['date']

                    # Get actual transcript content
                    transcript_data = self.get_transcript_content(ticker, quarter, year)

                    if transcript_data and transcript_data.get('content'):
                        # Ensure we have a valid date range (2005-2025)
                        try:
                            filing_date = pd.to_datetime(date)
                            if filing_date >= pd.to_datetime('2005-01-01') and filing_date <= pd.to_datetime('2025-12-31'):
                                all_transcripts.append({
                                    'ticker': ticker,
                                    'filing_date': date,
                                    'document_type': 'earnings_call',
                                    'text_content': transcript_data['content'],
                                    'quarter': quarter,
                                    'year': year,
                                    'url': None
                                })
                                ticker_transcripts += 1
                                total_transcripts += 1
                        except:
                            continue

                if ticker_transcripts == 0:
                    failed_tickers.append(ticker)

                # Rate limiting - more aggressive for comprehensive collection
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                failed_tickers.append(ticker)
                continue

        # Save results
        if all_transcripts:
            text_df = pd.DataFrame(all_transcripts)
            text_df['filing_date'] = pd.to_datetime(text_df['filing_date'])
            text_df.to_parquet(self.data_dir / "text_raw_comprehensive.parquet", index=False)

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
    membership_df = pd.read_parquet("data/sp500_quarterly_membership.parquet")
    all_tickers = sorted([t for t in membership_df['ticker'].unique() if t and pd.notna(t)])

    logger.info(f"Running ENHANCED transcript collection for {len(all_tickers)} S&P 500 companies")
    logger.info("Strategy: v4 API for metadata + v3 API with quarter/year for content")
    logger.info("Goal: Get ALL available transcripts from every quarter")

    # Run enhanced collection
    collector.download_comprehensive_transcripts(all_tickers)

if __name__ == "__main__":
    main()
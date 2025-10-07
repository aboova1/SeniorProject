#!/usr/bin/env python3
"""
Transcript Filter - Removes entries from quarters.parquet without earnings call transcripts
Filters the feature processor output to only include ticker-quarter combinations
that have corresponding earnings call transcripts within a reasonable time window.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranscriptFilter:
    def __init__(self, data_dir: str = "data"):
        """Initialize the filter

        Args:
            data_dir: Path to the data directory, relative to the script location
        """
        self.data_dir = Path(__file__).parent / data_dir

    def load_data(self):
        """Load quarters.parquet and text data"""
        logger.info("Loading data files...")

        # Load the feature processor output
        quarters_file = self.data_dir / "quarters.parquet"
        if not quarters_file.exists():
            raise FileNotFoundError("quarters.parquet not found. Run engineer_features.py first.")

        self.quarters_df = pd.read_parquet(quarters_file)
        logger.info(f"Loaded quarters data: {len(self.quarters_df)} observations")

        # Load transcript data
        text_file = self.data_dir / "text_raw_comprehensive.parquet"
        if not text_file.exists():
            raise FileNotFoundError("text_raw_comprehensive.parquet not found.")

        self.text_df = pd.read_parquet(text_file)
        logger.info(f"Loaded transcript data: {len(self.text_df)} transcripts")

        # Ensure date columns are datetime
        self.quarters_df['fiscal_quarter_end'] = pd.to_datetime(self.quarters_df['fiscal_quarter_end'])
        self.text_df['filing_date'] = pd.to_datetime(self.text_df['filing_date'])

    def match_transcripts_to_quarters(self, max_days_after: int = 90):
        """
        Match earnings call transcripts to fiscal quarters using vectorized operations

        Args:
            max_days_after: Maximum days after quarter end to look for transcript
        """
        logger.info(f"Matching transcripts to quarters (max {max_days_after} days after quarter end)...")

        # Create window bounds for each quarter
        self.quarters_df['window_start'] = self.quarters_df['fiscal_quarter_end']
        self.quarters_df['window_end'] = self.quarters_df['fiscal_quarter_end'] + timedelta(days=max_days_after)

        # Merge quarters with transcripts on ticker
        merged = self.quarters_df.merge(
            self.text_df[['ticker', 'filing_date', 'document_type']],
            on='ticker',
            how='left'
        )

        # Filter to transcripts within time window
        matched = merged[
            (merged['filing_date'] >= merged['window_start']) &
            (merged['filing_date'] <= merged['window_end'])
        ].copy()

        if len(matched) == 0:
            logger.warning("No quarters matched with transcripts!")
            return pd.DataFrame()

        # Keep earliest transcript per ticker-quarter
        matched['days_after_quarter'] = (matched['filing_date'] - matched['fiscal_quarter_end']).dt.days
        matched = matched.sort_values(['ticker', 'fiscal_quarter_end', 'days_after_quarter'])
        matched = matched.groupby(['ticker', 'fiscal_quarter_end']).first().reset_index()

        # Rename columns for clarity
        matched = matched.rename(columns={
            'filing_date': 'transcript_date',
            'document_type': 'transcript_type'
        })

        # Drop temporary window columns
        matched = matched.drop(['window_start', 'window_end'], axis=1)

        total_quarters = len(self.quarters_df)
        logger.info(f"Successfully matched {len(matched)} quarters with transcripts")
        logger.info(f"Filtered out {total_quarters - len(matched)} quarters without transcripts")
        logger.info(f"Retention rate: {len(matched)/total_quarters*100:.1f}%")

        return matched

    def analyze_transcript_coverage(self, filtered_df: pd.DataFrame):
        """Analyze the transcript coverage after filtering"""
        logger.info("Analyzing transcript coverage...")

        # Coverage by year
        filtered_df['year'] = filtered_df['fiscal_quarter_end'].dt.year
        coverage_by_year = filtered_df.groupby('year').agg({
            'ticker': 'nunique',
            'transcript_date': 'count',
            'days_after_quarter': 'mean'
        }).round(1)
        coverage_by_year.columns = ['unique_tickers', 'observations', 'avg_days_after_quarter']

        logger.info("Coverage by year:")
        for year, row in coverage_by_year.iterrows():
            logger.info(f"  {year}: {row['observations']:,} obs, {row['unique_tickers']} tickers, "
                       f"avg {row['avg_days_after_quarter']} days after quarter")

        # Coverage by ticker
        ticker_coverage = filtered_df.groupby('ticker').size().describe()
        logger.info(f"Quarters per ticker: mean={ticker_coverage['mean']:.1f}, "
                   f"median={ticker_coverage['50%']:.1f}, max={ticker_coverage['max']:.0f}")

        # Timing analysis
        timing_stats = filtered_df['days_after_quarter'].describe()
        logger.info(f"Days after quarter end: mean={timing_stats['mean']:.1f}, "
                   f"median={timing_stats['50%']:.1f}, max={timing_stats['max']:.0f}")

    def filter_quarters(self, max_days_after: int = 90, output_suffix: str = "_with_transcripts"):
        """
        Main filtering function

        Args:
            max_days_after: Maximum days after quarter end to look for transcript
            output_suffix: Suffix to add to output filename
        """
        logger.info("Starting transcript filtering process...")

        # Load data
        self.load_data()

        original_count = len(self.quarters_df)
        logger.info(f"Original dataset: {original_count:,} observations")
        logger.info(f"Date range: {self.quarters_df['fiscal_quarter_end'].min()} to {self.quarters_df['fiscal_quarter_end'].max()}")
        logger.info(f"Unique tickers: {self.quarters_df['ticker'].nunique()}")

        # Match transcripts to quarters
        filtered_df = self.match_transcripts_to_quarters(max_days_after=max_days_after)

        if len(filtered_df) == 0:
            logger.error("No data remaining after filtering!")
            return

        # Analyze coverage
        self.analyze_transcript_coverage(filtered_df)

        # Save filtered dataset
        output_file = self.data_dir / f"quarters{output_suffix}.parquet"
        filtered_df.to_parquet(output_file, index=False)

        logger.info("="*60)
        logger.info("FILTERING SUMMARY")
        logger.info("="*60)
        logger.info(f"Original observations: {original_count:,}")
        logger.info(f"Filtered observations: {len(filtered_df):,}")
        logger.info(f"Removed: {original_count - len(filtered_df):,} ({(original_count - len(filtered_df))/original_count*100:.1f}%)")
        logger.info(f"Retained: {len(filtered_df)/original_count*100:.1f}%")
        logger.info(f"Output saved to: {output_file}")
        logger.info("="*60)

        return filtered_df

def main():
    """Execute the transcript filter"""
    filter_tool = TranscriptFilter()

    # Filter with default settings (90 days after quarter end)
    filtered_df = filter_tool.filter_quarters()

    if filtered_df is not None and len(filtered_df) > 0:
        print(f"\n[SUCCESS] Created filtered dataset with {len(filtered_df):,} observations")
        print(f"[DATA] {filtered_df['ticker'].nunique()} companies with transcripts")
        print(f"[PERIOD] {filtered_df['fiscal_quarter_end'].min()} to {filtered_df['fiscal_quarter_end'].max()}")
        print(f"[TIMING] Average {filtered_df['days_after_quarter'].mean():.1f} days between quarter end and transcript")
    else:
        print("[FAILED] No data retained after filtering")

if __name__ == "__main__":
    main()
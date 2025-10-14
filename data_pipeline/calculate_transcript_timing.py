#!/usr/bin/env python3
"""
Calculate Company-Specific Transcript Timing
============================================

Analyzes historical transcript release patterns for each company
to determine optimal rebalance dates.

Output: transcript_timing_stats.parquet with per-company timing statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TranscriptTimingAnalyzer:
    """Analyze and calculate company-specific transcript release timing"""

    def __init__(self, data_dir: str = "data_pipeline/data"):
        self.data_dir = Path(data_dir)

    def calculate_company_delays(self) -> pd.DataFrame:
        """
        Calculate transcript release delays for each company.

        Returns DataFrame with columns:
        - ticker
        - mean_delay: Average days between fiscal quarter end and transcript
        - median_delay: Median delay
        - p75_delay: 75th percentile (conservative estimate)
        - p95_delay: 95th percentile (very conservative)
        - max_delay: Maximum observed delay
        - sample_size: Number of transcript observations
        """
        logger.info("Loading data...")

        # Load fundamentals (actual fiscal quarter ends)
        fundamentals = pd.read_parquet(self.data_dir / "fundamentals_quarterly.parquet")

        # Load transcripts (filing dates)
        text_df = pd.read_parquet(
            self.data_dir / "text_embeddings.parquet",
            columns=['ticker', 'filing_date']
        )

        logger.info(f"Loaded {fundamentals['ticker'].nunique()} companies from fundamentals")
        logger.info(f"Loaded {text_df['ticker'].nunique()} companies from transcripts")

        # Calculate delays for each company
        logger.info("Calculating transcript delays per company...")

        company_stats = []
        tickers = sorted(text_df['ticker'].unique())

        for i, ticker in enumerate(tickers):
            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i+1}/{len(tickers)}")

            # Get company data
            company_fund = fundamentals[fundamentals['ticker'] == ticker].sort_values('fiscal_quarter_end')
            company_text = text_df[text_df['ticker'] == ticker].sort_values('filing_date')

            if len(company_fund) == 0 or len(company_text) == 0:
                continue

            # Calculate delays between fiscal quarter ends and transcripts
            delays = []

            for _, transcript in company_text.iterrows():
                transcript_date = pd.to_datetime(transcript['filing_date'])

                # Find the most recent fiscal quarter end before this transcript
                prior_quarters = company_fund[
                    pd.to_datetime(company_fund['fiscal_quarter_end']) < transcript_date
                ]

                if not prior_quarters.empty:
                    latest_quarter = pd.to_datetime(prior_quarters.iloc[-1]['fiscal_quarter_end'])
                    gap = (transcript_date - latest_quarter).days

                    # Only include reasonable delays (0-120 days)
                    if 0 <= gap <= 120:
                        delays.append(gap)

            if len(delays) >= 3:  # Need at least 3 observations for meaningful stats
                company_stats.append({
                    'ticker': ticker,
                    'mean_delay': float(np.mean(delays)),
                    'median_delay': float(np.median(delays)),
                    'p75_delay': float(np.percentile(delays, 75)),
                    'p95_delay': float(np.percentile(delays, 95)),
                    'max_delay': float(np.max(delays)),
                    'min_delay': float(np.min(delays)),
                    'sample_size': len(delays)
                })

        df = pd.DataFrame(company_stats)
        logger.info(f"Calculated timing stats for {len(df)} companies")

        return df

    def calculate_recommended_rebalance_lags(self, timing_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate recommended rebalance lag for each company.

        Uses 75th percentile + 7 day buffer for safety.
        """
        timing_stats = timing_stats.copy()

        # Recommended lag: 75th percentile + 7 day buffer
        # This covers most transcripts while not being overly conservative
        timing_stats['recommended_lag_days'] = (timing_stats['p75_delay'] + 7).round().astype(int)

        # Alternative: Very conservative (95th percentile + 5 days)
        timing_stats['conservative_lag_days'] = (timing_stats['p95_delay'] + 5).round().astype(int)

        # Minimum lag: Use median for fast companies
        timing_stats['minimum_lag_days'] = timing_stats['median_delay'].round().astype(int)

        return timing_stats

    def save_and_summarize(self, timing_stats: pd.DataFrame):
        """Save timing statistics and print summary"""

        # Save to parquet
        output_path = self.data_dir / "transcript_timing_stats.parquet"
        timing_stats.to_parquet(output_path, index=False)
        logger.info(f"\nSaved transcript timing statistics to {output_path}")

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("TRANSCRIPT TIMING STATISTICS SUMMARY")
        logger.info("="*80)

        logger.info(f"\nTotal companies analyzed: {len(timing_stats)}")
        logger.info(f"Total transcript observations: {timing_stats['sample_size'].sum():,.0f}")

        logger.info("\nOverall delay statistics (days):")
        logger.info(f"  Mean across companies: {timing_stats['mean_delay'].mean():.1f}")
        logger.info(f"  Median across companies: {timing_stats['median_delay'].median():.1f}")

        logger.info("\nRecommended rebalance lags (p75 + 7 days):")
        logger.info(f"  Average: {timing_stats['recommended_lag_days'].mean():.1f} days")
        logger.info(f"  Median: {timing_stats['recommended_lag_days'].median():.0f} days")
        logger.info(f"  Min: {timing_stats['recommended_lag_days'].min()} days")
        logger.info(f"  Max: {timing_stats['recommended_lag_days'].max()} days")

        # Distribution of recommended lags
        logger.info("\nDistribution of recommended lags:")
        bins = [0, 20, 30, 40, 50, 60, 200]
        labels = ['0-20 days', '21-30 days', '31-40 days', '41-50 days', '51-60 days', '60+ days']

        for i, label in enumerate(labels):
            count = ((timing_stats['recommended_lag_days'] > bins[i]) &
                    (timing_stats['recommended_lag_days'] <= bins[i+1])).sum()
            pct = count / len(timing_stats) * 100
            logger.info(f"  {label}: {count} companies ({pct:.1f}%)")

        # Show fastest and slowest companies
        logger.info("\nFastest 10 companies (median delay):")
        fastest = timing_stats.nsmallest(10, 'median_delay')[['ticker', 'median_delay', 'recommended_lag_days']]
        for _, row in fastest.iterrows():
            logger.info(f"  {row['ticker']}: {row['median_delay']:.0f} days median, {row['recommended_lag_days']} recommended lag")

        logger.info("\nSlowest 10 companies (median delay):")
        slowest = timing_stats.nlargest(10, 'median_delay')[['ticker', 'median_delay', 'recommended_lag_days']]
        for _, row in slowest.iterrows():
            logger.info(f"  {row['ticker']}: {row['median_delay']:.0f} days median, {row['recommended_lag_days']} recommended lag")

        logger.info("="*80)

    def run(self):
        """Main execution"""
        logger.info("="*80)
        logger.info("CALCULATING COMPANY-SPECIFIC TRANSCRIPT TIMING")
        logger.info("="*80)

        # Calculate delays
        timing_stats = self.calculate_company_delays()

        if timing_stats.empty:
            logger.error("No timing statistics calculated!")
            return

        # Calculate recommended lags
        timing_stats = self.calculate_recommended_rebalance_lags(timing_stats)

        # Save and summarize
        self.save_and_summarize(timing_stats)

        logger.info("\nDone! Use these stats in engineer_features.py for company-specific rebalance dates.")


def main():
    analyzer = TranscriptTimingAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()

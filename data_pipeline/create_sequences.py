#!/usr/bin/env python3
"""
Sequence Creator - Generates overlapping 8-quarter sequences for model training
Takes quarters_with_transcripts.parquet and creates all possible 8-quarter sequences
with 1-quarter shifts (maximum overlap while maintaining uniqueness)

Now also splits sequences by TICKER into train/val/test (70/15/15) and saves 3 Parquet files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SequenceCreator:
    def __init__(self, data_dir: str = "data_pipeline/data", sequence_length: int = 8):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length

    def load_data(self):
        """Load the filtered quarters data"""
        logger.info("Loading filtered quarters data...")

        quarters_file = self.data_dir / "quarters_with_transcripts.parquet"
        if not quarters_file.exists():
            raise FileNotFoundError("quarters_with_transcripts.parquet not found. Run filter_by_transcripts.py first.")

        self.quarters_df = pd.read_parquet(quarters_file)
        logger.info(f"Loaded {len(self.quarters_df)} observations")
        logger.info(f"Date range: {self.quarters_df['fiscal_quarter_end'].min()} to {self.quarters_df['fiscal_quarter_end'].max()}")
        logger.info(f"Unique tickers: {self.quarters_df['ticker'].nunique()}")

        # Ensure sorted by ticker and date
        self.quarters_df = self.quarters_df.sort_values(['ticker', 'fiscal_quarter_end']).reset_index(drop=True)

    def create_sequences_for_ticker(self, ticker: str, ticker_data: pd.DataFrame) -> list:
        """
        Create all possible 8-quarter sequences for a single ticker
        Each sequence shifts by exactly 1 quarter (maximum data utilization)
        Only includes complete sequences with no missing data
        """
        sequences = []
        n_quarters = len(ticker_data)

        # Need at least 8 quarters to create a sequence
        if n_quarters < self.sequence_length:
            return sequences

        # Create overlapping sequences with 1-quarter shifts
        for start_idx in range(n_quarters - self.sequence_length + 1):
            end_idx = start_idx + self.sequence_length
            sequence_data = ticker_data.iloc[start_idx:end_idx].copy()

            # Verify this is actually 8 consecutive quarters (no gaps)
            dates = sequence_data['fiscal_quarter_end'].values
            quarters_diff = pd.to_datetime(dates[-1]) - pd.to_datetime(dates[0])
            # 8 quarters span ~7 quarter periods = 7 * 90 days = 630 days
            # Allow some flexibility for calendar differences (±60 days)
            if not (540 <= quarters_diff.days <= 720):
                logger.debug(f"Skipping sequence for {ticker} starting {dates[0]}: {quarters_diff.days} days span")
                continue

            # Only check for missing data in critical columns that are absolutely required
            critical_cols = ['ticker', 'fiscal_quarter_end', 'target_price_next_q', 'current_price', 'sector']
            critical_cols = [col for col in critical_cols if col in sequence_data.columns]

            if len(critical_cols) > 0 and sequence_data[critical_cols].isnull().any().any():
                missing_cols = sequence_data[critical_cols].columns[sequence_data[critical_cols].isnull().any()].tolist()
                logger.debug(f"Skipping sequence for {ticker} starting {dates[0]}: missing critical data in {missing_cols}")
                continue

            # Verify we have exactly 8 rows
            if len(sequence_data) != self.sequence_length:
                logger.debug(f"Skipping sequence for {ticker} starting {dates[0]}: only {len(sequence_data)} quarters")
                continue

            # Create sequence metadata
            sequence = {
                'ticker': ticker,
                'sequence_id': f"{ticker}_{start_idx}",
                'start_date': dates[0],
                'end_date': dates[-1],
                'start_idx': start_idx,
                'n_quarters_available': n_quarters,
                'sequence_data': sequence_data
            }

            sequences.append(sequence)

        return sequences

    def create_all_sequences(self):
        """Generate all possible 8-quarter sequences across all tickers"""
        logger.info(f"Creating {self.sequence_length}-quarter sequences with 1-quarter shifts...")
        logger.info("Filtering out sequences with gaps or missing data...")

        all_sequences = []
        tickers = self.quarters_df['ticker'].unique()
        total_attempted = 0

        for i, ticker in enumerate(tickers, 1):
            ticker_data = self.quarters_df[self.quarters_df['ticker'] == ticker].copy()
            n_quarters = len(ticker_data)

            # Count how many sequences we attempted for this ticker
            possible_sequences = max(0, n_quarters - self.sequence_length + 1)
            total_attempted += possible_sequences

            ticker_sequences = self.create_sequences_for_ticker(ticker, ticker_data)

            if ticker_sequences:
                all_sequences.extend(ticker_sequences)

            if i % 50 == 0:
                logger.info(f"Processed {i}/{len(tickers)} tickers, {len(all_sequences)} sequences created so far")

        logger.info(f"Created {len(all_sequences)} complete sequences from {total_attempted} attempted")
        if total_attempted > 0:
            logger.info(f"Filtered out {total_attempted - len(all_sequences)} incomplete sequences ({(total_attempted - len(all_sequences))/total_attempted*100:.1f}%)")

        return all_sequences

    def flatten_sequences(self, sequences: list) -> pd.DataFrame:
        """
        Convert list of sequences into a flat DataFrame
        Each row represents one quarter within a sequence
        """
        logger.info("Flattening sequences into training format...")

        flattened_rows = []

        for seq in sequences:
            sequence_data = seq['sequence_data']

            for quarter_idx, (_, row) in enumerate(sequence_data.iterrows()):
                flattened_row = {
                    'sequence_id': seq['sequence_id'],
                    'ticker': seq['ticker'],
                    'quarter_in_sequence': quarter_idx,  # 0-7 for 8 quarters
                    'sequence_start_date': seq['start_date'],
                    'sequence_end_date': seq['end_date'],
                    **row.to_dict()  # Include all original columns
                }
                flattened_rows.append(flattened_row)

        flattened_df = pd.DataFrame(flattened_rows)
        logger.info(f"Flattened to {len(flattened_df)} rows ({len(sequences)} sequences × ~{self.sequence_length} quarters)")

        return flattened_df

    def analyze_sequences(self, sequences_df: pd.DataFrame):
        """Analyze the created sequences"""
        logger.info("Analyzing sequence coverage...")

        n_sequences = sequences_df['sequence_id'].nunique()
        n_tickers = sequences_df['ticker'].nunique()

        # Sequences per ticker
        seq_per_ticker = sequences_df.groupby('ticker')['sequence_id'].nunique()
        logger.info(f"Total sequences: {n_sequences:,}")
        logger.info(f"Unique tickers: {n_tickers}")
        logger.info(f"Sequences per ticker: mean={seq_per_ticker.mean():.1f}, "
                   f"median={seq_per_ticker.median():.1f}, max={seq_per_ticker.max()}")

        # Date range
        logger.info(f"Date range: {sequences_df['fiscal_quarter_end'].min()} to {sequences_df['fiscal_quarter_end'].max()}")

        # Year distribution
        sequences_df = sequences_df.copy()
        sequences_df['year'] = pd.to_datetime(sequences_df['fiscal_quarter_end']).dt.year
        year_dist = sequences_df.groupby('year')['sequence_id'].nunique()
        logger.info("Sequences by year:")
        for year, count in year_dist.items():
            logger.info(f"  {year}: {count:,} sequences")

    def save_sequences(self, sequences_df: pd.DataFrame, output_file: str = "sequences_8q.parquet"):
        """Save sequences to parquet file (kept for optional full-dump)"""
        output_path = self.data_dir / output_file
        sequences_df.to_parquet(output_path, index=False)
        logger.info(f"Saved sequences to: {output_path}")
        return output_path

    # ---------- NEW: ticker-stratified split & save ----------
    def split_by_ticker_and_save(
        self,
        sequences_df: pd.DataFrame,
        train_p: float = 0.70,
        val_p: float = 0.15,
        test_p: float = 0.15,
        seed: int = 42,
        out_prefix: str = "sequences_8q"
    ):
        """
        Split by TICKER into train/val/test and save three Parquet files.
        Ensures all sequences for a given ticker go to the same split.
        """
        assert abs(train_p + val_p + test_p - 1.0) < 1e-9, "Splits must sum to 1.0"

        tickers = sequences_df['ticker'].dropna().unique()
        rng = np.random.RandomState(seed)
        rng.shuffle(tickers)

        n = len(tickers)
        n_train = int(round(train_p * n))
        n_val = int(round(val_p * n))
        # ensure all tickers assigned
        if n_train + n_val > n:
            n_train = int(np.floor(train_p * n))
            n_val = int(np.floor(val_p * n))
        n_test = n - n_train - n_val

        train_tickers = set(tickers[:n_train])
        val_tickers   = set(tickers[n_train:n_train + n_val])
        test_tickers  = set(tickers[n_train + n_val:])

        def _subset(df, tset): 
            return df[df['ticker'].isin(tset)].copy()

        train_df = _subset(sequences_df, train_tickers)
        val_df   = _subset(sequences_df, val_tickers)
        test_df  = _subset(sequences_df, test_tickers)

        # Save
        train_path = self.data_dir / f"{out_prefix}_train.parquet"
        val_path   = self.data_dir / f"{out_prefix}_val.parquet"
        test_path  = self.data_dir / f"{out_prefix}_test.parquet"

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        # Logging
        def _stats(name, df, tset):
            logger.info(
                f"{name}: {len(tset)} tickers | "
                f"{df['sequence_id'].nunique():,} sequences | "
                f"{len(df):,} rows"
            )

        logger.info("=== Split by ticker (seed=%d) ===", seed)
        logger.info(f"Total tickers: {n} | train/val/test target = {train_p:.2f}/{val_p:.2f}/{test_p:.2f}")
        _stats("TRAIN", train_df, train_tickers)
        _stats("VAL  ", val_df,   val_tickers)
        _stats("TEST ", test_df,  test_tickers)
        logger.info(f"Saved: {train_path.name}, {val_path.name}, {test_path.name}")

        return train_path, val_path, test_path

    def create_sequences(self, out_prefix: str = "sequences_8q"):
        """Main function to create and save sequences (now writes train/val/test splits)"""
        logger.info("="*60)
        logger.info("SEQUENCE CREATION STARTING")
        logger.info("="*60)

        # Load data
        self.load_data()

        # Create sequences
        sequences = self.create_all_sequences()

        if not sequences:
            logger.error("No sequences created!")
            return None

        # Flatten to DataFrame
        sequences_df = self.flatten_sequences(sequences)

        # Analyze
        self.analyze_sequences(sequences_df)

        # Split & save by ticker
        train_path, val_path, test_path = self.split_by_ticker_and_save(
            sequences_df,
            train_p=0.70,
            val_p=0.15,
            test_p=0.15,
            seed=42,
            out_prefix=out_prefix
        )

        logger.info("="*60)
        logger.info("SEQUENCE CREATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total sequences: {sequences_df['sequence_id'].nunique():,}")
        logger.info(f"Total rows: {len(sequences_df):,}")
        logger.info(f"Outputs: {train_path.name}, {val_path.name}, {test_path.name}")
        logger.info("="*60)

        return sequences_df

def main():
    """Execute sequence creation"""
    creator = SequenceCreator(sequence_length=8)
    sequences_df = creator.create_sequences(out_prefix="sequences_8q")

    if sequences_df is not None:
        print(f"\n[SUCCESS] Created {sequences_df['sequence_id'].nunique():,} sequences")
        print(f"[ROWS] {len(sequences_df):,} total rows (sequences × quarters)")
        print(f"[TICKERS] {sequences_df['ticker'].nunique()} unique companies")
        print("[READY] Wrote sequences_8q_train.parquet / _val.parquet / _test.parquet")
    else:
        print("[FAILED] No sequences created")

if __name__ == "__main__":
    main()

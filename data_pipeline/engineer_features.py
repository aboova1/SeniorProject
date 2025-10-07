#!/usr/bin/env python3
"""
Updated Feature Processor - Compatible with Precise S&P 500 Collector
Creates model-ready quarters.parquet from precise collector output
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _process_ticker_technical(args: tuple) -> list:
    """Process technical indicators for a single ticker - used for parallel processing"""
    ticker, ticker_prices, spy_returns_dict, ticker_quarters = args
    tech_indicators = []

    ticker_data = ticker_prices.sort_values('date').copy()

    if len(ticker_data) < 252:  # Need at least 1 year of data
        return []

    # Calculate returns
    ticker_data['returns'] = ticker_data['adj_close'].pct_change()

    # Merge with SPY for beta calculation - using dict to avoid passing large DataFrame
    ticker_data['spy_returns'] = ticker_data['date'].map(spy_returns_dict)
    merged_data = ticker_data.dropna(subset=['spy_returns'])

    if len(merged_data) < 252:
        return []

    # Calculate rolling indicators for each quarter end
    for quarter_end in ticker_quarters:
        quarter_date = pd.to_datetime(quarter_end)

        # Find closest trading day
        closest_idx = None
        min_diff = float('inf')

        for idx, row in merged_data.iterrows():
            diff = abs((row['date'] - quarter_date).days)
            if diff < min_diff and diff <= 10:
                min_diff = diff
                closest_idx = idx

        if closest_idx is None:
            continue

        row_pos = merged_data.index.get_loc(closest_idx)

        if row_pos < 252:
            continue

        # 12-month momentum
        current_price = merged_data.iloc[row_pos]['adj_close']
        year_ago_price = merged_data.iloc[row_pos-252]['adj_close']
        momentum_12m = (current_price / year_ago_price) - 1

        # 60-day volatility
        if row_pos >= 60:
            vol_window = merged_data.iloc[row_pos-60:row_pos]['returns'].dropna()
            vol_60d = vol_window.std() * np.sqrt(252) if len(vol_window) > 10 else 0.0
        else:
            vol_60d = 0.0

        # 1-year beta
        beta_window = merged_data.iloc[max(0, row_pos-252):row_pos]
        common_data = beta_window[['returns', 'spy_returns']].dropna()

        if len(common_data) > 50:
            stock_ret = common_data['returns']
            spy_ret = common_data['spy_returns']
            spy_var = spy_ret.var()
            beta_1y = np.cov(stock_ret, spy_ret)[0, 1] / spy_var if spy_var > 0 else 1.0
        else:
            beta_1y = 1.0

        tech_indicators.append({
            'ticker': ticker,
            'fiscal_quarter_end': quarter_date,
            'momentum_12m': momentum_12m,
            'vol_60d': vol_60d,
            'beta_1y': beta_1y
        })

    return tech_indicators

def _process_ticker_target_prices(args: tuple) -> list:
    """Process target prices for a single ticker - used for parallel processing"""
    ticker, ticker_prices, ticker_quarters = args
    target_prices = []

    ticker_data = ticker_prices.sort_values('date').copy()
    ticker_data['date'] = pd.to_datetime(ticker_data['date'])

    if len(ticker_data) < 126:  # Need at least 6 months of data
        return []

    ticker_quarters = pd.to_datetime(ticker_quarters)

    # First, find the matched price for each quarter (do this once for all quarters)
    # This ensures Q(n).target_price_next_q == Q(n+1).current_price
    quarter_prices = {}
    for quarter_date in ticker_quarters:
        quarter_df = pd.DataFrame({'quarter_date': [quarter_date]})
        matched = pd.merge_asof(
            quarter_df,
            ticker_data[['date', 'adj_close']],
            left_on='quarter_date',
            right_on='date',
            direction='nearest',
            tolerance=pd.Timedelta(days=10)
        )

        if pd.notna(matched['adj_close'].iloc[0]):
            quarter_prices[quarter_date] = matched['adj_close'].iloc[0]

    # Now create target prices using the pre-matched prices
    for i, quarter_date in enumerate(ticker_quarters):
        if quarter_date not in quarter_prices:
            continue

        current_price = quarter_prices[quarter_date]

        # Get next quarter's price (if available)
        if i + 1 < len(ticker_quarters):
            next_quarter_date = ticker_quarters[i + 1]

            if next_quarter_date not in quarter_prices:
                continue

            next_quarter_price = quarter_prices[next_quarter_date]
        else:
            # No next quarter available
            continue

        target_prices.append({
            'ticker': ticker,
            'fiscal_quarter_end': quarter_date,
            'current_price': current_price,
            'target_price_next_q': next_quarter_price
        })

    return target_prices

class UpdatedFeatureProcessor:
    def __init__(self, data_dir: str = "data_pipeline/data", max_workers: int = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        # Use 12 workers for parallel processing
        import multiprocessing
        self.max_workers = max_workers or 20

    def load_data(self):
        """Load all downloaded data files from precise collector"""
        logger.info("Loading data files from precise collector...")

        # Check which files exist
        required_files = [
            "sp500_quarterly_membership.parquet",
            "prices_daily.parquet",
            "fundamentals_quarterly.parquet",
            "earnings_data.parquet"
        ]

        missing_files = []
        for file in required_files:
            if not (self.data_dir / file).exists():
                missing_files.append(file)

        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            raise FileNotFoundError(f"Missing files: {missing_files}")

        # Load all data
        self.membership_df = pd.read_parquet(self.data_dir / "sp500_quarterly_membership.parquet")
        self.prices_df = pd.read_parquet(self.data_dir / "prices_daily.parquet")
        self.fundamentals_df = pd.read_parquet(self.data_dir / "fundamentals_quarterly.parquet")
        self.earnings_df = pd.read_parquet(self.data_dir / "earnings_data.parquet")

        logger.info(f"Loaded membership data: {len(self.membership_df)} ticker-quarter combinations")
        logger.info(f"Loaded price data: {len(self.prices_df)} daily observations")
        logger.info(f"Loaded fundamentals: {len(self.fundamentals_df)} quarterly observations")
        logger.info(f"Loaded earnings: {len(self.earnings_df)} earnings records")

    def get_sector_mapping(self):
        """Create sector mapping from available data"""
        logger.info("Creating sector mapping...")

        # Try to get sector info from fundamentals data or create dummy mapping
        if 'sector' in self.fundamentals_df.columns:
            sector_map = self.fundamentals_df.groupby('ticker')['sector'].first().to_dict()
        else:
            # Create dummy sector mapping
            unique_tickers = self.membership_df['ticker'].unique()
            sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer Discretionary',
                      'Communication Services', 'Industrials', 'Consumer Staples',
                      'Energy', 'Utilities', 'Real Estate', 'Materials']

            sector_map = {}
            for i, ticker in enumerate(unique_tickers):
                sector_map[ticker] = sectors[i % len(sectors)]

        logger.info(f"Created sector mapping for {len(sector_map)} tickers")
        return sector_map

    def calculate_technical_indicators(self, prices_df: pd.DataFrame, membership_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum, volatility, and beta for S&P 500 members in parallel"""
        logger.info("Calculating technical indicators...")

        # Get SPY data for beta calculation
        spy_data = prices_df[prices_df['ticker'] == 'SPY'].sort_values('date').copy()
        if len(spy_data) == 0:
            logger.error("No SPY data found! Please run the data collector first to download SPY data.")
            raise ValueError("SPY data is required for technical indicators calculation")

        spy_data['spy_returns'] = spy_data['adj_close'].pct_change()
        # Convert to dict for efficient serialization
        spy_returns_dict = dict(zip(spy_data['date'], spy_data['spy_returns']))

        # Get unique tickers from membership
        sp500_tickers = [t for t in membership_df['ticker'].unique() if t != 'SPY']
        logger.info(f"Calculating technical indicators for {len(sp500_tickers)} S&P 500 companies")
        logger.info(f"Using {self.max_workers} parallel workers")

        # Pre-filter data per ticker to avoid sending large DataFrames to workers
        ticker_data_dict = {}
        ticker_quarters_dict = {}
        for ticker in sp500_tickers:
            ticker_data_dict[ticker] = prices_df[prices_df['ticker'] == ticker].copy()
            ticker_quarters_dict[ticker] = membership_df[membership_df['ticker'] == ticker]['quarter_end'].values

        # Process tickers in batches to avoid memory exhaustion
        batch_size = 50  # Process 50 tickers at a time
        tech_indicators = []

        for batch_start in range(0, len(sp500_tickers), batch_size):
            batch_end = min(batch_start + batch_size, len(sp500_tickers))
            batch_tickers = sp500_tickers[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(sp500_tickers) + batch_size - 1)//batch_size} "
                       f"(tickers {batch_start+1}-{batch_end})")

            # Prepare arguments for this batch
            task_args = [
                (ticker, ticker_data_dict[ticker], spy_returns_dict, ticker_quarters_dict[ticker])
                for ticker in batch_tickers
            ]

            # Process batch in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks for this batch
                futures = {executor.submit(_process_ticker_technical, args): args[0] for args in task_args}

                # Collect results
                completed = 0
                for future in as_completed(futures):
                    ticker = futures[future]
                    completed += 1

                    if completed % 10 == 0:
                        global_completed = batch_start + completed
                        logger.info(f"Technical indicators progress: {global_completed}/{len(sp500_tickers)}")

                    try:
                        result = future.result()
                        if result:
                            tech_indicators.extend(result)
                    except Exception as e:
                        logger.error(f"Error processing {ticker}: {e}")

        tech_df = pd.DataFrame(tech_indicators)
        logger.info(f"Calculated technical indicators for {len(tech_df)} ticker-quarter observations")
        return tech_df

    def calculate_fundamental_features(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fundamental ratios and features"""
        logger.info("Calculating fundamental features...")

        features_df = fundamentals_df.copy()

        # Calculate financial ratios with safe division
        def safe_divide(num, denom, default=np.nan):
            return np.where((denom != 0) & pd.notna(denom) & pd.notna(num), num / denom, default)

        # Earnings yield (EPS / Price per share, approximated)
        features_df['earnings_yield'] = safe_divide(features_df['eps'],
                                                   features_df['market_cap'] / 1000000000)  # Convert to per-share basis

        # Margins
        features_df['operating_margin'] = safe_divide(features_df['operating_income'], features_df['revenue'])
        features_df['fcf_margin'] = safe_divide(features_df['free_cash_flow'], features_df['revenue'])

        # Balance sheet ratios
        features_df['current_ratio'] = safe_divide(features_df['current_assets'], features_df['current_liabilities'])

        # Ensure we have the key ratios that might be in the data already
        if 'debt_to_equity' not in features_df.columns:
            features_df['debt_to_equity'] = safe_divide(features_df['total_debt'], features_df['stockholders_equity'])

        if 'roe' not in features_df.columns:
            features_df['roe'] = safe_divide(features_df['net_income'], features_df['stockholders_equity'])

        # Calculate year-over-year growth rates
        features_df = features_df.sort_values(['ticker', 'fiscal_quarter_end'])

        features_df['revenue_yoy'] = features_df.groupby('ticker')['revenue'].pct_change(periods=4)
        features_df['eps_yoy'] = features_df.groupby('ticker')['eps'].pct_change(periods=4)

        # Calculate accruals
        features_df['accruals_scaled'] = safe_divide(
            (features_df['net_income'] - features_df['operating_cash_flow']),
            features_df['total_assets']
        )

        # Clean infinite values
        numeric_cols = ['earnings_yield', 'operating_margin', 'roe', 'revenue_yoy', 'eps_yoy',
                       'debt_to_equity', 'current_ratio', 'accruals_scaled', 'fcf_margin']

        for col in numeric_cols:
            if col in features_df.columns:
                features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)

        logger.info("Fundamental features calculated")
        return features_df

    def load_precomputed_embeddings(self) -> pd.DataFrame:
        """Load precomputed FinBERT embeddings and apply PCA"""
        logger.info("Loading precomputed FinBERT embeddings...")

        embeddings_path = self.data_dir / "text_embeddings.parquet"

        # Check if embeddings file exists
        if not embeddings_path.exists():
            logger.warning(f"Embeddings file not found at {embeddings_path}")
            logger.warning("Please run finbert_embeddings.py first to generate embeddings")
            logger.warning("Falling back to empty embeddings")
            return pd.DataFrame()

        # Load embeddings
        embeddings_df = pd.read_parquet(embeddings_path)
        logger.info(f"Loaded {len(embeddings_df)} precomputed embeddings")

        # Extract embedding columns (emb_000 to emb_XXX - auto-detect dimension)
        embedding_cols = [col for col in embeddings_df.columns if col.startswith('emb_')]
        available_embedding_cols = sorted(embedding_cols)

        if len(available_embedding_cols) == 0:
            logger.warning("No embedding columns found in file")
            return pd.DataFrame()

        # Extract embeddings matrix
        embeddings_matrix = embeddings_df[available_embedding_cols].values

        # Check for and handle NaN values - remove rows with NaN
        nan_mask = np.isnan(embeddings_matrix).any(axis=1)
        if nan_mask.any():
            n_nan = nan_mask.sum()
            logger.warning(f"Found {n_nan} embeddings with NaN values, removing these rows")
            valid_mask = ~nan_mask
            embeddings_df = embeddings_df[valid_mask].copy()
            embeddings_matrix = embeddings_matrix[valid_mask]
            logger.info(f"Remaining embeddings after removing NaN: {len(embeddings_df)}")

        # Apply PCA to reduce to 32 dimensions
        if len(embeddings_df) == 0:
            logger.warning("No valid embeddings remaining after NaN removal")
            return pd.DataFrame()

        n_components = min(32, len(embeddings_df), len(available_embedding_cols))
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings_matrix)

        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

        # Pad with zeros if we have fewer than 32 components
        if n_components < 32:
            padding = np.zeros((len(embeddings_df), 32 - n_components))
            reduced_embeddings = np.hstack([reduced_embeddings, padding])

        # Create DataFrame with reduced embeddings
        text_features_df = embeddings_df[['ticker', 'filing_date', 'document_type']].copy()

        # Add 32 PCA features
        for j in range(32):
            text_features_df[f'txt_{j+1:02d}'] = reduced_embeddings[:, j]

        logger.info(f"Processed embeddings: {len(text_features_df)} documents with 32 PCA features")

        return text_features_df

    def calculate_target_prices(self, prices_df: pd.DataFrame, membership_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate next quarter share price for S&P 500 members (parallelized)"""
        logger.info("Calculating target share prices...")

        sp500_tickers = [t for t in membership_df['ticker'].unique() if t != 'SPY']
        logger.info(f"Processing target prices for {len(sp500_tickers)} tickers")
        logger.info(f"Using {self.max_workers} parallel workers")

        # Pre-filter data per ticker
        ticker_data_dict = {}
        ticker_quarters_dict = {}
        for ticker in sp500_tickers:
            ticker_data_dict[ticker] = prices_df[prices_df['ticker'] == ticker].copy()
            ticker_quarters_dict[ticker] = membership_df[membership_df['ticker'] == ticker]['quarter_end'].values

        # Process tickers in batches
        batch_size = 50
        target_prices = []

        for batch_start in range(0, len(sp500_tickers), batch_size):
            batch_end = min(batch_start + batch_size, len(sp500_tickers))
            batch_tickers = sp500_tickers[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(sp500_tickers) + batch_size - 1)//batch_size} "
                       f"(tickers {batch_start+1}-{batch_end})")

            # Prepare arguments for this batch
            task_args = [
                (ticker, ticker_data_dict[ticker], ticker_quarters_dict[ticker])
                for ticker in batch_tickers
            ]

            # Process batch in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_process_ticker_target_prices, args): args[0] for args in task_args}

                # Collect results
                completed = 0
                for future in as_completed(futures):
                    ticker = futures[future]
                    completed += 1

                    if completed % 10 == 0:
                        global_completed = batch_start + completed
                        logger.info(f"Target prices progress: {global_completed}/{len(sp500_tickers)}")

                    try:
                        result = future.result()
                        if result:
                            target_prices.extend(result)
                    except Exception as e:
                        logger.error(f"Error processing {ticker}: {e}")

        target_df = pd.DataFrame(target_prices)
        logger.info(f"Calculated target prices for {len(target_df)} observations")
        return target_df

    def create_master_table(self) -> pd.DataFrame:
        """Create the master quarters.parquet table"""
        logger.info("Creating master quarters table...")

        # Load data
        self.load_data()

        # Get sector mapping
        sector_map = self.get_sector_mapping()

        # Calculate all features
        fundamental_features = self.calculate_fundamental_features(self.fundamentals_df)
        tech_indicators = self.calculate_technical_indicators(self.prices_df, self.membership_df)
        text_features = self.load_precomputed_embeddings()
        target_prices = self.calculate_target_prices(self.prices_df, self.membership_df)

        # Start with S&P 500 membership as base
        master_df = self.membership_df.copy()
        master_df = master_df.rename(columns={'quarter_end': 'fiscal_quarter_end'})

        # Add sector information
        master_df['sector'] = master_df['ticker'].map(sector_map)

        # Merge fundamental features
        master_df = master_df.merge(
            fundamental_features,
            on=['ticker', 'fiscal_quarter_end'],
            how='left'
        )

        # Merge technical indicators
        master_df = master_df.merge(
            tech_indicators,
            on=['ticker', 'fiscal_quarter_end'],
            how='left'
        )

        # Merge earnings data
        if len(self.earnings_df) > 0:
            earnings_features = self.earnings_df.copy()
            earnings_features['fiscal_quarter_end'] = pd.to_datetime(earnings_features['date'])

            # Get quarterly earnings surprise
            earnings_quarterly = earnings_features.groupby(['ticker', 'fiscal_quarter_end']).agg({
                'earnings_surprise_pct': 'last'
            }).reset_index()

            master_df = master_df.merge(earnings_quarterly, on=['ticker', 'fiscal_quarter_end'], how='left')

        # Merge text features (match transcripts to quarters based on filing date)
        if len(text_features) > 0:
            # Match each quarter to the closest transcript within a reasonable time window
            # Transcripts typically occur 30-60 days after quarter end
            logger.info("Matching transcripts to quarters...")

            # Ensure filing_date is datetime
            text_features['filing_date'] = pd.to_datetime(text_features['filing_date'])

            # Map filing date to the quarter it refers to
            # Earnings calls typically happen 30-45 days after quarter end
            # Subtract 45 days from filing date and round to nearest quarter end
            text_features['inferred_quarter'] = text_features['filing_date'] - pd.Timedelta(days=45)
            text_features['inferred_quarter'] = text_features['inferred_quarter'].dt.to_period('Q').dt.to_timestamp(how='end')
            # Normalize to start of day (00:00:00) to match membership data
            text_features['inferred_quarter'] = text_features['inferred_quarter'].dt.normalize()

            # Prepare text features for merge
            text_for_merge = text_features.copy()
            text_for_merge['fiscal_quarter_end'] = text_for_merge['inferred_quarter']

            # Keep only the closest transcript per ticker-quarter
            # Filter to transcripts that occur 0-90 days after the quarter end
            text_for_merge['days_after_quarter'] = (text_for_merge['filing_date'] - text_for_merge['fiscal_quarter_end']).dt.days
            text_for_merge = text_for_merge[
                (text_for_merge['days_after_quarter'] >= 0) &
                (text_for_merge['days_after_quarter'] <= 90)
            ]
            text_for_merge = text_for_merge.sort_values(['ticker', 'fiscal_quarter_end', 'days_after_quarter'])
            text_for_merge = text_for_merge.groupby(['ticker', 'fiscal_quarter_end']).first().reset_index()

            text_cols = ['ticker', 'fiscal_quarter_end'] + [f'txt_{i+1:02d}' for i in range(32)]
            available_text_cols = [col for col in text_cols if col in text_for_merge.columns]

            if len(available_text_cols) > 2:
                master_df = master_df.merge(text_for_merge[available_text_cols], on=['ticker', 'fiscal_quarter_end'], how='left')
                matched_count = master_df[[f'txt_{i+1:02d}' for i in range(32) if f'txt_{i+1:02d}' in master_df.columns]].notna().any(axis=1).sum()
                logger.info(f"Matched {matched_count} quarters with transcript embeddings")

        # Merge target prices
        master_df = master_df.merge(target_prices, on=['ticker', 'fiscal_quarter_end'], how='left')

        # Add rebalance date
        master_df['rebalance_date'] = pd.to_datetime(master_df['fiscal_quarter_end']) + pd.Timedelta(days=45)

        # Remove rows without target values
        initial_rows = len(master_df)
        master_df = master_df.dropna(subset=['target_price_next_q'])
        logger.info(f"Removed {initial_rows - len(master_df)} rows without target prices")

        # Filter to 2005 onwards
        master_df = master_df[master_df['fiscal_quarter_end'] >= '2005-01-01']

        # Sort by ticker and date
        master_df = master_df.sort_values(['ticker', 'fiscal_quarter_end'])

        # Save master table
        master_df.to_parquet(self.data_dir / "quarters.parquet", index=False)

        logger.info("="*60)
        logger.info("MASTER TABLE SUMMARY")
        logger.info("="*60)
        logger.info(f"Final dataset shape: {master_df.shape}")
        logger.info(f"Date range: {master_df['fiscal_quarter_end'].min()} to {master_df['fiscal_quarter_end'].max()}")
        logger.info(f"Unique tickers: {master_df['ticker'].nunique()}")
        logger.info(f"Quarters covered: {master_df['fiscal_quarter_end'].nunique()}")

        # Expected vs actual
        expected_obs = master_df['ticker'].nunique() * master_df['fiscal_quarter_end'].nunique()
        logger.info(f"Theoretical maximum: {expected_obs:,} observations")
        logger.info(f"Actual observations: {len(master_df):,}")
        logger.info(f"Data completeness: {len(master_df)/expected_obs*100:.1f}%")

        # Check for missing data
        missing_summary = master_df.isnull().sum()
        high_missing = missing_summary[missing_summary > len(master_df) * 0.1]
        if len(high_missing) > 0:
            logger.info(f"Columns with >10% missing data: {dict(high_missing)}")

        logger.info("="*60)

        return master_df

def main():
    """Execute the updated feature processor"""
    processor = UpdatedFeatureProcessor()
    master_df = processor.create_master_table()

    print(f"\n[SUCCESS] Created quarters.parquet with {len(master_df):,} observations")
    print(f"[DATA] {master_df['ticker'].nunique()} companies x {master_df['fiscal_quarter_end'].nunique()} quarters")
    print(f"[PERIOD] {master_df['fiscal_quarter_end'].min()} to {master_df['fiscal_quarter_end'].max()}")

if __name__ == "__main__":
    main()
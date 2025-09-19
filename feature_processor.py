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
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UpdatedFeatureProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load all downloaded data files from precise collector"""
        logger.info("Loading data files from precise collector...")

        # Check which files exist
        required_files = [
            "sp500_quarterly_membership.parquet",
            "prices_daily.parquet",
            "fundamentals_quarterly.parquet",
            "earnings_data.parquet",
            "text_raw.parquet"
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
        self.text_df = pd.read_parquet(self.data_dir / "text_raw.parquet")

        logger.info(f"Loaded membership data: {len(self.membership_df)} ticker-quarter combinations")
        logger.info(f"Loaded price data: {len(self.prices_df)} daily observations")
        logger.info(f"Loaded fundamentals: {len(self.fundamentals_df)} quarterly observations")
        logger.info(f"Loaded earnings: {len(self.earnings_df)} earnings records")
        logger.info(f"Loaded text: {len(self.text_df)} text documents")

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
        """Calculate momentum, volatility, and beta for S&P 500 members only"""
        logger.info("Calculating technical indicators...")

        tech_indicators = []

        # Get SPY data for beta calculation
        spy_data = prices_df[prices_df['ticker'] == 'SPY'].sort_values('date').copy()
        if len(spy_data) == 0:
            logger.warning("No SPY data found, using market average for beta calculation")
            # Use a large cap stock as proxy if SPY not available
            proxy_ticker = prices_df['ticker'].iloc[0]
            spy_data = prices_df[prices_df['ticker'] == proxy_ticker].sort_values('date').copy()

        spy_data['spy_returns'] = spy_data['adj_close'].pct_change()

        # Get unique tickers from membership
        sp500_tickers = membership_df['ticker'].unique()
        logger.info(f"Calculating technical indicators for {len(sp500_tickers)} S&P 500 companies")

        for i, ticker in enumerate(sp500_tickers):
            if ticker == 'SPY':
                continue

            if i % 100 == 0:
                logger.info(f"Technical indicators progress: {i+1}/{len(sp500_tickers)} ({ticker})")

            ticker_data = prices_df[prices_df['ticker'] == ticker].sort_values('date').copy()

            if len(ticker_data) < 252:  # Need at least 1 year of data
                continue

            # Calculate returns
            ticker_data['returns'] = ticker_data['adj_close'].pct_change()

            # Merge with SPY for beta calculation
            merged_data = ticker_data.merge(spy_data[['date', 'spy_returns']], on='date', how='inner')

            if len(merged_data) < 252:
                continue

            # Get quarters when this ticker was in S&P 500
            ticker_quarters = membership_df[membership_df['ticker'] == ticker]['quarter_end'].values

            # Calculate rolling indicators for each quarter end when ticker was in S&P 500
            for quarter_end in ticker_quarters:
                # Find the closest trading day to quarter end
                quarter_date = pd.to_datetime(quarter_end)
                closest_idx = None
                min_diff = float('inf')

                for idx, row in merged_data.iterrows():
                    diff = abs((row['date'] - quarter_date).days)
                    if diff < min_diff and diff <= 10:  # Within 10 days of quarter end
                        min_diff = diff
                        closest_idx = idx

                if closest_idx is None:
                    continue

                row_pos = merged_data.index.get_loc(closest_idx)

                if row_pos < 252:  # Need at least 1 year of history
                    continue

                # 12-month momentum (252 trading days)
                current_price = merged_data.iloc[row_pos]['adj_close']
                year_ago_price = merged_data.iloc[row_pos-252]['adj_close']
                momentum_12m = (current_price / year_ago_price) - 1

                # 60-day volatility
                if row_pos >= 60:
                    vol_window = merged_data.iloc[row_pos-60:row_pos]['returns'].dropna()
                    if len(vol_window) > 10:
                        vol_60d = vol_window.std() * np.sqrt(252)
                    else:
                        vol_60d = 0.0
                else:
                    vol_60d = 0.0

                # 1-year beta
                beta_window = merged_data.iloc[max(0, row_pos-252):row_pos]
                common_data = beta_window[['returns', 'spy_returns']].dropna()

                if len(common_data) > 50:
                    stock_ret = common_data['returns']
                    spy_ret = common_data['spy_returns']
                    spy_var = spy_ret.var()

                    if spy_var > 0:
                        beta_1y = np.cov(stock_ret, spy_ret)[0, 1] / spy_var
                    else:
                        beta_1y = 1.0
                else:
                    beta_1y = 1.0

                tech_indicators.append({
                    'ticker': ticker,
                    'fiscal_quarter_end': quarter_date,
                    'momentum_12m': momentum_12m,
                    'vol_60d': vol_60d,
                    'beta_1y': beta_1y
                })

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

    def process_text_embeddings(self, text_df: pd.DataFrame) -> pd.DataFrame:
        """Process text with dummy FinBERT embeddings and PCA"""
        logger.info("Processing text embeddings...")

        if len(text_df) == 0:
            logger.warning("No text data available, creating dummy embeddings")
            return pd.DataFrame()

        # Create dummy embeddings for each text document
        text_embeddings = []

        # Set seed for reproducible embeddings
        np.random.seed(42)

        for _, row in text_df.iterrows():
            # Create dummy 768-dimensional embedding based on ticker
            ticker_hash = hash(row['ticker']) % 1000
            embedding = np.random.normal(ticker_hash/1000, 0.1, 768)

            text_embeddings.append({
                'ticker': row['ticker'],
                'filing_date': row['filing_date'],
                'document_type': row['document_type'],
                'embedding': embedding
            })

        if len(text_embeddings) == 0:
            return pd.DataFrame()

        # Apply PCA to reduce to 64 dimensions
        embeddings_matrix = np.stack([item['embedding'] for item in text_embeddings])

        # Handle case where we have fewer samples than components
        n_components = min(64, len(text_embeddings))
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings_matrix)

        # Pad with zeros if we have fewer than 64 components
        if n_components < 64:
            padding = np.zeros((len(text_embeddings), 64 - n_components))
            reduced_embeddings = np.hstack([reduced_embeddings, padding])

        # Create DataFrame with reduced embeddings
        text_features = []
        for i, item in enumerate(text_embeddings):
            row_data = {
                'ticker': item['ticker'],
                'filing_date': item['filing_date'],
                'document_type': item['document_type']
            }

            # Add 64 text features
            for j in range(64):
                row_data[f'txt_{j+1:02d}'] = reduced_embeddings[i, j]

            text_features.append(row_data)

        text_features_df = pd.DataFrame(text_features)
        logger.info(f"Processed text embeddings for {len(text_features_df)} documents")

        return text_features_df

    def calculate_target_returns(self, prices_df: pd.DataFrame, membership_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate next quarter excess returns for S&P 500 members"""
        logger.info("Calculating target returns...")

        # Get SPY returns
        spy_data = prices_df[prices_df['ticker'] == 'SPY'].sort_values('date')
        if len(spy_data) == 0:
            logger.warning("No SPY data found for target calculation")
            return pd.DataFrame()

        spy_data['spy_log_return'] = np.log(spy_data['adj_close'] / spy_data['adj_close'].shift(1))

        target_returns = []
        sp500_tickers = membership_df['ticker'].unique()

        for ticker in sp500_tickers:
            if ticker == 'SPY':
                continue

            ticker_data = prices_df[prices_df['ticker'] == ticker].sort_values('date')
            if len(ticker_data) < 126:  # Need at least 6 months of data
                continue

            ticker_data['log_return'] = np.log(ticker_data['adj_close'] / ticker_data['adj_close'].shift(1))

            # Merge with SPY data
            merged_data = ticker_data.merge(spy_data[['date', 'spy_log_return']], on='date', how='inner')

            # Get quarters when this ticker was in S&P 500
            ticker_quarters = membership_df[membership_df['ticker'] == ticker]['quarter_end'].values

            for quarter_end in ticker_quarters:
                quarter_date = pd.to_datetime(quarter_end)

                # Find data point closest to quarter end
                quarter_data = merged_data[merged_data['date'] <= quarter_date]
                if len(quarter_data) == 0:
                    continue

                quarter_idx = len(quarter_data) - 1

                # Check if we have enough future data (next quarter)
                if quarter_idx + 63 >= len(merged_data):  # 63 trading days ≈ 1 quarter
                    continue

                # Calculate next quarter returns
                next_quarter_stock = merged_data.iloc[quarter_idx+1:quarter_idx+64]['log_return'].sum()
                next_quarter_spy = merged_data.iloc[quarter_idx+1:quarter_idx+64]['spy_log_return'].sum()

                if pd.notna(next_quarter_stock) and pd.notna(next_quarter_spy):
                    excess_return = next_quarter_stock - next_quarter_spy

                    target_returns.append({
                        'ticker': ticker,
                        'fiscal_quarter_end': quarter_date,
                        'target_excess_return_next_q': excess_return
                    })

        target_df = pd.DataFrame(target_returns)
        logger.info(f"Calculated target returns for {len(target_df)} observations")
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
        text_features = self.process_text_embeddings(self.text_df)
        target_returns = self.calculate_target_returns(self.prices_df, self.membership_df)

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

        # Merge text features (most recent per ticker)
        if len(text_features) > 0:
            text_quarterly = text_features.sort_values(['ticker', 'filing_date']).groupby('ticker').tail(1)
            text_cols = ['ticker'] + [f'txt_{i+1:02d}' for i in range(64)]
            available_text_cols = [col for col in text_cols if col in text_quarterly.columns]

            if len(available_text_cols) > 1:
                master_df = master_df.merge(text_quarterly[available_text_cols], on='ticker', how='left')

        # Merge target returns
        master_df = master_df.merge(target_returns, on=['ticker', 'fiscal_quarter_end'], how='left')

        # Add rebalance date
        master_df['rebalance_date'] = pd.to_datetime(master_df['fiscal_quarter_end']) + pd.Timedelta(days=45)

        # Remove rows without target values
        initial_rows = len(master_df)
        master_df = master_df.dropna(subset=['target_excess_return_next_q'])
        logger.info(f"Removed {initial_rows - len(master_df)} rows without target returns")

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

    print(f"\n✅ Created quarters.parquet with {len(master_df):,} observations")
    print(f"📊 {master_df['ticker'].nunique()} companies × {master_df['fiscal_quarter_end'].nunique()} quarters")
    print(f"📅 Period: {master_df['fiscal_quarter_end'].min()} to {master_df['fiscal_quarter_end'].max()}")

if __name__ == "__main__":
    main()
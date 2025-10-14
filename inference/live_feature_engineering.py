#!/usr/bin/env python3
"""
Lightweight Feature Engineering for Live Predictions
Calculates the essential features needed by the model
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveFeatureEngineer:
    """Calculate features for live prediction data"""

    def __init__(self, use_transcript_embeddings: bool = True):
        self.use_transcript_embeddings = use_transcript_embeddings
        self.transcript_embedder = None

        if use_transcript_embeddings:
            try:
                from live_transcript_embeddings_v2 import LiveTranscriptEmbedder
                logger.info("Initializing transcript embedder...")
                self.transcript_embedder = LiveTranscriptEmbedder()
            except Exception as e:
                logger.warning(f"Could not load transcript embedder: {e}")
                logger.warning("Will use zero embeddings instead")
                self.use_transcript_embeddings = False

    def calculate_fundamental_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fundamental financial ratios"""
        df = df.copy()

        # Profitability ratios
        df['gross_margin'] = np.where(df['revenue'] != 0, df['grossProfit'] / df['revenue'], 0)
        df['operating_margin'] = np.where(df['revenue'] != 0, df['operatingIncome'] / df['revenue'], 0)
        df['net_margin'] = np.where(df['revenue'] != 0, df['netIncome'] / df['revenue'], 0)

        # Efficiency ratios
        df['roa'] = np.where(df['totalAssets'] != 0, df['netIncome'] / df['totalAssets'], 0)
        df['roe'] = np.where(df['totalEquity'] != 0, df['netIncome'] / df['totalEquity'], 0)

        # Leverage ratios
        df['debt_to_equity'] = np.where(df['totalEquity'] != 0, df['totalDebt'] / df['totalEquity'], 0)
        df['debt_to_assets'] = np.where(df['totalAssets'] != 0, df['totalDebt'] / df['totalAssets'], 0)

        # Cash flow ratios
        df['fcf_margin'] = np.where(df['revenue'] != 0, df['freeCashFlow'] / df['revenue'], 0)
        df['ocf_to_revenue'] = np.where(df['revenue'] != 0, df['operatingCashFlow'] / df['revenue'], 0)

        return df

    def calculate_growth_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate year-over-year and quarter-over-quarter growth rates"""
        df = df.copy()
        df = df.sort_values('fiscal_quarter_end').reset_index(drop=True)

        metrics = ['revenue', 'netIncome', 'operatingCashFlow', 'eps']

        for metric in metrics:
            if metric in df.columns:
                # YoY growth (compare to 4 quarters ago)
                df[f'{metric}_yoy_growth'] = df[metric].pct_change(periods=4)

                # QoQ growth (compare to previous quarter)
                df[f'{metric}_qoq_growth'] = df[metric].pct_change(periods=1)

        return df

    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features"""
        df = df.copy()
        df = df.sort_values('fiscal_quarter_end').reset_index(drop=True)

        if 'current_price' in df.columns:
            # Price momentum
            df['price_return_1q'] = df['current_price'].pct_change(periods=1)
            df['price_return_2q'] = df['current_price'].pct_change(periods=2)
            df['price_return_4q'] = df['current_price'].pct_change(periods=4)

            # Price volatility (rolling std)
            df['price_volatility_4q'] = df['current_price'].rolling(window=4, min_periods=2).std()

        return df

    def calculate_valuation_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate valuation metrics"""
        df = df.copy()

        # PE ratio (should already exist, but calculate if missing)
        if 'peRatio' not in df.columns and 'current_price' in df.columns and 'eps' in df.columns:
            df['peRatio'] = np.where(df['eps'] != 0, df['current_price'] / df['eps'], np.nan)

        # Price to sales
        if 'priceToSalesRatio' not in df.columns and 'marketCap' in df.columns and 'revenue' in df.columns:
            df['priceToSalesRatio'] = np.where(df['revenue'] != 0, df['marketCap'] / df['revenue'], np.nan)

        # EV to sales
        if 'evToSales' not in df.columns and 'enterpriseValue' in df.columns and 'revenue' in df.columns:
            df['evToSales'] = np.where(df['revenue'] != 0, df['enterpriseValue'] / df['revenue'], np.nan)

        return df

    def calculate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling statistics"""
        df = df.copy()
        df = df.sort_values('fiscal_quarter_end').reset_index(drop=True)

        metrics = ['revenue', 'netIncome', 'gross_margin', 'net_margin']

        for metric in metrics:
            if metric in df.columns:
                # Rolling mean (4 quarters)
                df[f'{metric}_ma4'] = df[metric].rolling(window=4, min_periods=2).mean()

                # Rolling std (4 quarters)
                df[f'{metric}_std4'] = df[metric].rolling(window=4, min_periods=2).std()

        return df

    def create_exact_model_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the exact 62 features the model expects, matching training data column names
        """
        df = df.copy()
        df = df.sort_values('fiscal_quarter_end').reset_index(drop=True)

        # Map raw data to model feature names
        df['net_income'] = df['netIncome']
        df['operating_income'] = df['operatingIncome']
        df['gross_profit'] = df['grossProfit']
        df['ebitda'] = df['operating_income']  # Approximation
        df['stockholders_equity'] = df['totalEquity']
        df['total_debt'] = df['totalDebt']  # Add this mapping
        df['total_assets'] = df['totalAssets']  # Add this mapping
        df['free_cash_flow'] = df['freeCashFlow']
        df['operating_cash_flow'] = df['operatingCashFlow']
        df['capital_expenditure'] = df['capitalExpenditure']
        df['pe_ratio'] = df['peRatio']
        df['market_cap'] = df['marketCap']
        df['enterprise_value'] = df['enterpriseValue']

        # Estimate current assets/liabilities if missing
        df['current_assets'] = df.get('cashAndCashEquivalents', df['total_assets'] * 0.3)
        df['current_liabilities'] = df['totalLiabilities'] * 0.4

        # Calculate ratios
        df['current_ratio'] = np.where(df['current_liabilities'] != 0,
                                       df['current_assets'] / df['current_liabilities'], 1.0)
        df['debt_to_equity'] = np.where(df['stockholders_equity'] != 0,
                                        df['total_debt'] / df['stockholders_equity'], 0)
        df['roe'] = np.where(df['stockholders_equity'] != 0,
                            df['net_income'] / df['stockholders_equity'], 0)
        df['operating_margin'] = np.where(df['revenue'] != 0,
                                         df['operating_income'] / df['revenue'], 0)
        df['earnings_yield'] = np.where(df['pe_ratio'] != 0, 1.0 / df['pe_ratio'], 0)
        df['fcf_margin'] = np.where(df['revenue'] != 0,
                                    df['free_cash_flow'] / df['revenue'], 0)

        # Growth rates
        df['revenue_yoy'] = df['revenue'].pct_change(periods=4)
        df['eps_yoy'] = df['eps'].pct_change(periods=4)

        # Accruals
        df['accruals_scaled'] = np.where(df['total_assets'] != 0,
                                         (df['net_income'] - df['operating_cash_flow']) / df['total_assets'], 0)

        # Price features
        if 'current_price' in df.columns:
            df['momentum_12m'] = df['current_price'].pct_change(periods=4)
            df['vol_60d'] = df['current_price'].rolling(window=4, min_periods=2).std()
            df['beta_1y'] = 1.0
        else:
            df['momentum_12m'] = 0
            df['vol_60d'] = 0
            df['beta_1y'] = 1.0

        # Earnings surprise (no estimates available)
        df['earnings_surprise_pct'] = 0

        # Text embeddings - fetch real transcripts if enabled
        if self.use_transcript_embeddings and self.transcript_embedder is not None:
            try:
                logger.info("Fetching transcripts and generating embeddings...")
                ticker = df['ticker'].iloc[0]
                quarter_dates = df['fiscal_quarter_end'].tolist()

                # Fetch transcripts for all quarters
                transcripts = self.transcript_embedder.fetch_transcripts_for_quarters(ticker, quarter_dates)

                # Generate embeddings
                quarter_date_strs = [qd.strftime('%Y-%m-%d') for qd in quarter_dates]
                embedding_df = self.transcript_embedder.process_transcripts_to_features(
                    transcripts, quarter_date_strs
                )

                # Add embeddings to dataframe
                for col in embedding_df.columns:
                    df[col] = embedding_df[col].values

                logger.info("✓ Real transcript embeddings added")

            except Exception as e:
                logger.warning(f"Error generating transcript embeddings: {e}")
                logger.warning("Falling back to zero embeddings")
                for i in range(1, 33):
                    df[f'txt_{i:02d}'] = 0.0
        else:
            # No transcript embeddings - set to 0
            for i in range(1, 33):
                df[f'txt_{i:02d}'] = 0.0

        # Fill NaN and inf
        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps to the dataframe

        Args:
            df: DataFrame with raw quarterly data

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features for live data...")

        # Apply all feature engineering steps
        df = self.calculate_fundamental_ratios(df)
        df = self.calculate_growth_rates(df)
        df = self.calculate_price_features(df)
        df = self.calculate_valuation_metrics(df)
        df = self.calculate_rolling_features(df)

        # Create the exact 62 features the model expects
        df = self.create_exact_model_features(df)

        # Fill infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        logger.info(f"Feature engineering complete. Shape: {df.shape}")

        return df


# Test
if __name__ == "__main__":
    from live_data_fetcher import LiveDataFetcher

    # Fetch data for AAPL
    fetcher = LiveDataFetcher()
    data = fetcher.fetch_complete_data('AAPL', quarters=7)

    if data is not None:
        print("\n=== Before Feature Engineering ===")
        print(f"Columns: {len(data.columns)}")
        print(data.columns.tolist())

        # Engineer features
        engineer = LiveFeatureEngineer()
        data_with_features = engineer.engineer_features(data)

        print("\n=== After Feature Engineering ===")
        print(f"Columns: {len(data_with_features.columns)}")
        print(data_with_features.columns.tolist())

        # Show some features
        print("\n=== Sample Features ===")
        print(data_with_features[['ticker', 'fiscal_quarter_end', 'revenue', 'net_margin', 'roe', 'revenue_yoy_growth']].tail())

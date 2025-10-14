#!/usr/bin/env python3
"""Test transcript embeddings in feature engineering"""

from live_data_fetcher import LiveDataFetcher
from live_feature_engineering import LiveFeatureEngineer

print('Fetching data for AAPL...')
fetcher = LiveDataFetcher()
data = fetcher.fetch_complete_data('AAPL', quarters=7)

print('\nEngineering features WITH transcript embeddings...')
engineer = LiveFeatureEngineer(use_transcript_embeddings=True)
data_with_features = engineer.engineer_features(data)

print('\nChecking txt_ columns...')
txt_cols = [c for c in data_with_features.columns if c.startswith('txt_')]
print(f'Found {len(txt_cols)} text embedding columns')

if txt_cols:
    print(f'\nSample txt_01 values (first 3 rows):')
    print(data_with_features['txt_01'].values[:3])

    nonzero_count = sum(1 for col in txt_cols if (data_with_features[col] != 0).any())
    print(f'\nNon-zero embedding columns: {nonzero_count}/{len(txt_cols)}')

    if nonzero_count > 0:
        print('SUCCESS: Real transcript embeddings are being used!')
    else:
        print('WARNING: All embeddings are zero')

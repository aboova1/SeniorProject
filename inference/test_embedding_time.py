#!/usr/bin/env python3
"""Test how long it takes to generate embeddings"""

from live_transcript_embeddings import LiveTranscriptEmbedder
import time

embedder = LiveTranscriptEmbedder()

# Fetch one transcript
print('Fetching AAPL Q2 2025 transcript...')
transcript = embedder.fetch_transcript('AAPL', 2025, 2)
print(f'Length: {len(transcript)} chars')

# Time the embedding generation
print('\nGenerating embedding (this should take a few seconds)...')
start = time.time()
embedding = embedder.generate_embedding(transcript)
elapsed = time.time() - start

print(f'Embedding generated in {elapsed:.2f} seconds')
print(f'Embedding shape: {embedding.shape}')
print(f'First 10 values: {embedding[:10]}')
print(f'Sum of all values: {embedding.sum():.4f}')
print(f'Non-zero values: {(embedding != 0).sum()}/{len(embedding)}')

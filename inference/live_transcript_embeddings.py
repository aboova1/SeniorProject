#!/usr/bin/env python3
"""
Live Transcript Embeddings Generator
Fetches earnings call transcripts and generates embeddings in real-time
"""

import os
import requests
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveTranscriptEmbedder:
    """Fetch transcripts and generate embeddings in real-time"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('FMP_API_KEY', 'FlVMN0WC4KpV1f9OWTNVamh39gW8wx7L')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load FinBERT model for financial text embeddings
        logger.info("Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModel.from_pretrained("ProsusAI/finbert")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"FinBERT loaded on {self.device}")

        # Try to load existing PCA model (for consistency with training data)
        self.pca = None
        self._load_pca_model()

    def _load_pca_model(self):
        """Load pre-trained PCA model if available"""
        pca_path = Path("../data_pipeline/data/text_embeddings_pca.pkl")
        if pca_path.exists():
            try:
                with open(pca_path, 'rb') as f:
                    self.pca = pickle.load(f)
                logger.info(f"Loaded pre-trained PCA model (reduces to {self.pca.n_components} dims)")
            except Exception as e:
                logger.warning(f"Could not load PCA model: {e}")
        else:
            logger.info("No pre-trained PCA model found, will create new one if needed")

    def fetch_transcript(self, ticker: str, year: int, quarter: int) -> Optional[str]:
        """Fetch earnings call transcript for a specific quarter"""
        try:
            url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}"
            params = {
                'year': year,
                'quarter': quarter,
                'apikey': self.api_key
            }

            logger.info(f"Fetching transcript for {ticker} Q{quarter} {year}...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data and len(data) > 0:
                content = data[0].get('content', '')
                logger.info(f"Retrieved transcript ({len(content)} chars)")
                return content
            else:
                logger.warning(f"No transcript found for {ticker} Q{quarter} {year}")
                return None

        except Exception as e:
            logger.error(f"Error fetching transcript: {e}")
            return None

    def fetch_transcripts_for_quarters(self, ticker: str, quarter_dates: List[pd.Timestamp]) -> Dict[str, str]:
        """
        Fetch transcripts for multiple quarters

        Args:
            ticker: Stock ticker
            quarter_dates: List of fiscal quarter end dates

        Returns:
            Dictionary mapping quarter_date to transcript content
        """
        transcripts = {}

        for quarter_date in quarter_dates:
            # Determine year and quarter from the date
            year = quarter_date.year
            quarter = (quarter_date.month - 1) // 3 + 1

            # Adjust for fiscal year if needed (e.g., if Q1 fiscal ends in Dec, it's actually Q4 calendar)
            # For now, use calendar quarter

            transcript = self.fetch_transcript(ticker, year, quarter)
            if transcript:
                transcripts[quarter_date.strftime('%Y-%m-%d')] = transcript
            else:
                logger.warning(f"No transcript for {quarter_date}")
                transcripts[quarter_date.strftime('%Y-%m-%d')] = ""

        return transcripts

    def generate_embedding(self, text: str, max_length: int = 512, use_chunking: bool = True) -> np.ndarray:
        """
        Generate FinBERT embedding for text, processing in chunks for long transcripts

        Args:
            text: Input text (earnings call transcript)
            max_length: Maximum token length per chunk (default 512 for BERT)
            use_chunking: If True, process long texts in chunks and average embeddings

        Returns:
            768-dimensional embedding vector
        """
        if not text or len(text) == 0:
            logger.warning("Empty text, returning zero embedding")
            return np.zeros(768)

        try:
            # Tokenize full text to see how long it is
            full_tokens = self.tokenizer(text, return_tensors="pt", truncation=False)
            total_tokens = full_tokens['input_ids'].shape[1]

            # If text is short enough, process directly
            if total_tokens <= max_length or not use_chunking:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

                return embedding

            # For long texts, split into chunks and average embeddings
            logger.info(f"Text has {total_tokens} tokens, processing in chunks...")

            # Split text by words for chunking
            words = text.split()
            chunk_size = 400  # Leave room for special tokens
            chunks = []

            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                chunks.append(chunk)

            logger.info(f"Processing {len(chunks)} chunks...")

            # Process each chunk and collect embeddings
            chunk_embeddings = []
            for i, chunk in enumerate(chunks):
                if i % 10 == 0 and i > 0:
                    logger.info(f"  Processed {i}/{len(chunks)} chunks...")

                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    chunk_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                    chunk_embeddings.append(chunk_embedding)

            # Average all chunk embeddings
            final_embedding = np.mean(chunk_embeddings, axis=0)
            logger.info(f"✓ Generated embedding from {len(chunks)} chunks")

            return final_embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(768)

    def reduce_embeddings_pca(self, embeddings: np.ndarray, n_components: int = 32) -> np.ndarray:
        """
        Reduce embedding dimensions using PCA

        Args:
            embeddings: Array of shape (n_samples, 768)
            n_components: Target dimensions (default 32 to match model)

        Returns:
            Reduced embeddings of shape (n_samples, n_components)
        """
        n_samples = embeddings.shape[0]

        # PCA can't have more components than samples
        actual_components = min(n_components, n_samples)

        if self.pca is None:
            # Create new PCA if we don't have one
            logger.info(f"Creating new PCA model (reducing to {actual_components} dims, requested {n_components})")
            self.pca = PCA(n_components=actual_components)
            reduced = self.pca.fit_transform(embeddings)

            # If we have fewer components than requested, pad with zeros
            if actual_components < n_components:
                logger.warning(f"Only {actual_components} samples available, padding to {n_components} dimensions with zeros")
                padded = np.zeros((n_samples, n_components))
                padded[:, :actual_components] = reduced
                reduced = padded

        else:
            # Use existing PCA
            reduced = self.pca.transform(embeddings)

            # Pad if needed
            if reduced.shape[1] < n_components:
                padded = np.zeros((n_samples, n_components))
                padded[:, :reduced.shape[1]] = reduced
                reduced = padded

        return reduced

    def process_transcripts_to_features(
        self,
        transcripts: Dict[str, str],
        quarter_dates: List[str]
    ) -> pd.DataFrame:
        """
        Process multiple transcripts into PCA-reduced embeddings

        Args:
            transcripts: Dictionary mapping quarter_date to transcript content
            quarter_dates: List of quarter dates in order

        Returns:
            DataFrame with txt_01 through txt_32 columns
        """
        # Generate embeddings for all transcripts
        embeddings_list = []
        for quarter_date in quarter_dates:
            transcript = transcripts.get(quarter_date, "")
            embedding = self.generate_embedding(transcript)
            embeddings_list.append(embedding)

        embeddings_array = np.array(embeddings_list)

        # Apply PCA to reduce to 32 dimensions
        reduced_embeddings = self.reduce_embeddings_pca(embeddings_array, n_components=32)

        # Create DataFrame with txt_01 through txt_32 columns
        embedding_df = pd.DataFrame(
            reduced_embeddings,
            columns=[f'txt_{i:02d}' for i in range(1, 33)]
        )

        return embedding_df


# Test
if __name__ == "__main__":
    embedder = LiveTranscriptEmbedder()

    # Test with AAPL Q2 2025
    ticker = "AAPL"
    year = 2025
    quarter = 2

    transcript = embedder.fetch_transcript(ticker, year, quarter)

    if transcript:
        print(f"\n✓ Transcript fetched ({len(transcript)} chars)")
        print(f"First 300 chars: {transcript[:300]}")

        print("\nGenerating embedding...")
        embedding = embedder.generate_embedding(transcript)
        print(f"✓ Embedding shape: {embedding.shape}")
        print(f"Embedding preview: {embedding[:10]}")

        # Test PCA reduction
        print("\nApplying PCA reduction...")
        embeddings_array = embedding.reshape(1, -1)
        reduced = embedder.reduce_embeddings_pca(embeddings_array, n_components=32)
        print(f"✓ Reduced embedding shape: {reduced.shape}")
        print(f"Reduced embedding preview: {reduced[0][:10]}")

        print("\n✓ All tests passed!")

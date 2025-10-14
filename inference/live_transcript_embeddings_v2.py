#!/usr/bin/env python3
"""
Live Transcript Embeddings Generator - V2
Matches the exact implementation from generate_text_embeddings.py
Uses 4-bit quantization and proper chunking with overlap
"""

import os
import requests
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from sklearn.decomposition import PCA
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveTranscriptEmbedder:
    """Fetch transcripts and generate embeddings using exact pipeline approach"""

    def __init__(self, api_key: str = None, model_name: str = "intfloat/e5-mistral-7b-instruct"):
        self.api_key = api_key or os.environ.get('FMP_API_KEY', 'FlVMN0WC4KpV1f9OWTNVamh39gW8wx7L')
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_e5_model = "e5" in model_name.lower()

        logger.info(f"Loading {model_name} with 4-bit quantization...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Use 4-bit quantization - EXACTLY like your pipeline
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
            )
            logger.info("✓ Loaded model with 4-bit quantization (NF4)")
        except Exception as e:
            logger.warning(f"4-bit quantization failed: {e}, trying float16")
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                self.model.to(self.device)
            logger.info("✓ Loaded model with float16")

        self.model.eval()
        logger.info(f"Model ready on {self.device}")

        # Load PCA if available
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

    def fetch_transcript(self, ticker: str, year: int, quarter: int) -> Optional[str]:
        """Fetch earnings call transcript for a specific quarter"""
        try:
            url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}"
            params = {
                'year': year,
                'quarter': quarter,
                'apikey': self.api_key
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data and len(data) > 0:
                content = data[0].get('content', '')
                return content
            else:
                return None

        except Exception as e:
            logger.error(f"Error fetching transcript: {e}")
            return None

    def fetch_transcripts_for_quarters(self, ticker: str, quarter_dates: List[pd.Timestamp]) -> Dict[str, str]:
        """Fetch transcripts for multiple quarters"""
        transcripts = {}

        for quarter_date in quarter_dates:
            year = quarter_date.year
            quarter = (quarter_date.month - 1) // 3 + 1

            transcript = self.fetch_transcript(ticker, year, quarter)
            if transcript:
                transcripts[quarter_date.strftime('%Y-%m-%d')] = transcript
            else:
                logger.warning(f"No transcript for {quarter_date}")
                transcripts[quarter_date.strftime('%Y-%m-%d')] = ""

        return transcripts

    def chunk_text(self, text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
        """
        Split long text into overlapping chunks - EXACTLY like your pipeline

        Args:
            text: Input text to chunk
            max_length: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks

        Returns:
            List of text chunks
        """
        # Tokenize the full text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # If text fits in one chunk, return it
        if len(tokens) <= max_length - 2:  # -2 for [CLS] and [SEP]
            return [text]

        # Create overlapping chunks
        chunks = []
        stride = max_length - overlap - 2  # -2 for special tokens

        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + max_length - 2]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            # Stop if we've covered all tokens
            if i + max_length - 2 >= len(tokens):
                break

        return chunks

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate FinBERT embedding for text - EXACTLY like your pipeline
        Handles long documents via chunking with overlap

        Returns:
            768-dimensional embedding vector (averaged over chunks)
        """
        embedding_size = 4096 if self.is_e5_model else 768
        
        if not text or len(text) == 0:
            logger.warning("Empty text, returning zero embedding")
            return np.zeros(embedding_size)

        try:
            # For E5 models, prepend instruction
            if self.is_e5_model:
                text = f"Instruct: Retrieve relevant financial information\nQuery: {text}"

            # Chunk the text with overlap
            chunks = self.chunk_text(text, max_length=512, overlap=50)

            if len(chunks) > 1:
                logger.info(f"Processing {len(chunks)} chunks with 50-token overlap...")

            # Generate embeddings for each chunk
            chunk_embeddings = []

            # Clear GPU cache before starting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                for i, chunk in enumerate(chunks):
                    if i > 0 and i % 10 == 0:
                        logger.info(f"  Processed {i}/{len(chunks)} chunks...")
                        # Clear GPU cache periodically
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # Tokenize - batch size = 1 (smallest possible)
                    inputs = self.tokenizer(
                        chunk,
                        return_tensors='pt',
                        truncation=True,
                        padding=True,
                        max_length=512
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Get model output
                    outputs = self.model(**inputs)

                    # Use [CLS] token for FinBERT, mean pooling for E5
                    if self.is_e5_model:
                        embedding = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
                    else:
                        embedding = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
                    chunk_embeddings.append(embedding[0])

                    # Clear intermediate tensors immediately
                    del outputs, inputs

            # Final GPU cache clear
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Average embeddings across chunks - EXACTLY like your pipeline
            final_embedding = np.mean(chunk_embeddings, axis=0)

            if len(chunks) > 1:
                logger.info(f"✓ Generated embedding from {len(chunks)} chunks")

            return final_embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(768)

    def reduce_embeddings_pca(self, embeddings: np.ndarray, n_components: int = 32) -> np.ndarray:
        """Reduce embedding dimensions using PCA"""
        n_samples = embeddings.shape[0]
        actual_components = min(n_components, n_samples)

        if self.pca is None:
            logger.info(f"Creating new PCA model (reducing to {actual_components} dims)")
            self.pca = PCA(n_components=actual_components)
            reduced = self.pca.fit_transform(embeddings)

            if actual_components < n_components:
                logger.warning(f"Only {actual_components} samples, padding to {n_components} dims")
                padded = np.zeros((n_samples, n_components))
                padded[:, :actual_components] = reduced
                reduced = padded
        else:
            reduced = self.pca.transform(embeddings)
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
        """Process multiple transcripts into PCA-reduced embeddings"""
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
    import time

    embedder = LiveTranscriptEmbedder()

    # Test with AAPL Q2 2025
    ticker = "AAPL"
    year = 2025
    quarter = 2

    print(f"\nFetching transcript for {ticker} Q{quarter} {year}...")
    transcript = embedder.fetch_transcript(ticker, year, quarter)

    if transcript:
        print(f"Transcript length: {len(transcript)} chars")
        print(f"Transcript words: ~{len(transcript.split())} words")

        print("\nGenerating embedding with chunking and overlap...")
        start = time.time()
        embedding = embedder.generate_embedding(transcript)
        elapsed = time.time() - start

        print(f"\nEmbedding generated in {elapsed:.2f} seconds")
        print(f"Embedding shape: {embedding.shape}")
        print(f"First 10 values: {embedding[:10]}")
        print(f"Sum: {embedding.sum():.4f}")
        print(f"Non-zero values: {(embedding != 0).sum()}/{len(embedding)}")

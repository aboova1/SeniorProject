#!/usr/bin/env python3
"""
FinBERT Embeddings Generator
Computes real FinBERT embeddings for earnings call transcripts
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from tqdm import tqdm
import time
import warnings
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranscriptDataset(Dataset):
    """Dataset for batching transcripts with pre-chunked text"""
    def __init__(self, dataframe, tokenizer, is_e5_model, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.is_e5_model = is_e5_model
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text_content']

        # Prepend instruction for E5 models
        if self.is_e5_model and text and not pd.isna(text):
            text = f"Instruct: Retrieve relevant financial information\nQuery: {text}"

        return {
            'text': text if text and not pd.isna(text) else "",
            'ticker': row['ticker'],
            'filing_date': row['filing_date'],
            'document_type': row['document_type'],
            'quarter': row.get('quarter', None),
            'year': row.get('year', None),
            'idx': idx
        }

def collate_chunks(batch, tokenizer, max_length=512):
    """Collate function that tokenizes and creates chunks from multiple transcripts"""
    all_chunk_data = []

    for item in batch:
        text = item['text']
        if not text:
            # Empty text case
            all_chunk_data.append({
                'metadata': item,
                'chunk_idx': 0,
                'num_chunks': 1,
                'is_empty': True
            })
            continue

        # Tokenize and chunk
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= max_length - 2:
            # Single chunk
            all_chunk_data.append({
                'text': text,
                'metadata': item,
                'chunk_idx': 0,
                'num_chunks': 1,
                'is_empty': False
            })
        else:
            # Multiple chunks
            overlap = 50
            stride = max_length - overlap - 2
            chunks = []

            for i in range(0, len(tokens), stride):
                chunk_tokens = tokens[i:i + max_length - 2]
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)

                if i + max_length - 2 >= len(tokens):
                    break

            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunk_data.append({
                    'text': chunk_text,
                    'metadata': item,
                    'chunk_idx': chunk_idx,
                    'num_chunks': len(chunks),
                    'is_empty': False
                })

    # Now tokenize all chunks together for efficient GPU processing
    texts_to_tokenize = [item['text'] for item in all_chunk_data if not item.get('is_empty', False)]

    if texts_to_tokenize:
        tokenized = tokenizer(
            texts_to_tokenize,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=max_length
        )
    else:
        tokenized = None

    return {
        'tokenized': tokenized,
        'chunk_data': all_chunk_data
    }

class FinBERTEmbeddings:
    def __init__(self, data_dir: str = "data_pipeline/data", model_name: str = "intfloat/e5-mistral-7b-instruct"):
        """
        Initialize financial text embeddings generator

        Args:
            data_dir: Directory containing text data
            model_name: HuggingFace model name (default: intfloat/e5-mistral-7b-instruct for Fin-E5)
                       Alternative: "ProsusAI/finbert" for original FinBERT
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_e5_model = "e5" in model_name.lower()

        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model: {model_name}")
        if self.is_e5_model:
            logger.info("Using Fin-E5 (state-of-the-art for financial embeddings as of 2025)")

        # Load tokenizer and model with optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Use 4-bit quantization to reduce VRAM usage even further
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
            logger.info("Loaded model with 4-bit quantization (NF4)")
        except Exception as e:
            logger.warning(f"4-bit quantization failed: {e}, trying 8-bit")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
                logger.info("Loaded model with 8-bit quantization")
            except Exception as e:
                logger.warning(f"8-bit quantization failed: {e}, loading in float16")
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                if not torch.cuda.is_available():
                    self.model.to(self.device)

        self.model.eval()

        # Enable gradient checkpointing to reduce memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Enable GPU optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            # Clear any cached memory
            torch.cuda.empty_cache()

        # Compile model for faster inference (PyTorch 2.0+)
        # Note: torch.compile requires Triton which is not available on Windows
        if torch.cuda.is_available() and hasattr(torch, 'compile'):
            try:
                import platform
                if platform.system() != 'Windows':
                    logger.info("Compiling model with torch.compile()...")
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                    logger.info("Model compiled successfully")
                else:
                    logger.info("Skipping torch.compile on Windows (Triton not available)")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
                logger.info("Continuing without compilation")

        logger.info("Model loaded successfully")

    def verify_embeddings_integrity(self, embeddings_path: Path) -> dict:
        """
        Verify the integrity of the embeddings file
        
        Args:
            embeddings_path: Path to the embeddings parquet file
            
        Returns:
            Dictionary with verification results
        """
        if not embeddings_path.exists():
            return {"status": "file_not_found", "message": "Embeddings file does not exist"}
        
        try:
            df = pd.read_parquet(embeddings_path)
            
            # Check for required columns
            required_cols = ['ticker', 'filing_date']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {"status": "missing_columns", "message": f"Missing required columns: {missing_cols}"}
            
            # Check for embedding columns
            embedding_cols = [col for col in df.columns if col.startswith('emb_')]
            if not embedding_cols:
                return {"status": "no_embeddings", "message": "No embedding columns found"}
            
            # Check for duplicates
            duplicates = df.duplicated(subset=['ticker', 'filing_date']).sum()
            
            # Check for missing values in embeddings
            embedding_missing = df[embedding_cols].isnull().sum().sum()
            
            return {
                "status": "valid",
                "total_records": len(df),
                "unique_tickers": df['ticker'].nunique(),
                "date_range": (df['filing_date'].min(), df['filing_date'].max()),
                "embedding_dimension": len(embedding_cols),
                "duplicates": duplicates,
                "missing_embeddings": embedding_missing,
                "file_size_mb": embeddings_path.stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Error reading file: {str(e)}"}

    def chunk_text(self, text: str, max_length: int = 512, overlap: int = 50) -> list:
        """
        Split long text into overlapping chunks that fit within token limit

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

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text (handles long documents via chunking)

        Args:
            text: Input text to embed

        Returns:
            Embedding vector (4096-dim for E5, 768-dim for FinBERT) averaged over chunks
        """
        # Determine embedding size based on model
        embedding_size = 4096 if self.is_e5_model else 768

        if not text or pd.isna(text):
            return np.zeros(embedding_size)

        # For E5 models, prepend instruction as per model documentation
        if self.is_e5_model:
            text = f"Instruct: Retrieve relevant financial information\nQuery: {text}"

        # Chunk the text
        chunks = self.chunk_text(text)

        if len(chunks) == 0:
            return np.zeros(embedding_size)

        # Generate embeddings for each chunk
        chunk_embeddings = []

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for chunk in chunks:
                # Tokenize
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

                # Use [CLS] token embedding or mean pooling depending on model
                if self.is_e5_model:
                    # E5 uses last hidden state with mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
                else:
                    # FinBERT uses [CLS] token
                    embedding = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()

                chunk_embeddings.append(embedding[0])

        # Average embeddings across chunks
        final_embedding = np.mean(chunk_embeddings, axis=0)

        return final_embedding

    def compute_embeddings_batch(self, texts: list, batch_size: int = 8) -> np.ndarray:
        """
        Compute embeddings for a batch of texts (more efficient for short texts)

        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once

        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        embeddings = []

        # Prepend instruction for E5 models
        if self.is_e5_model:
            texts = [f"Instruct: Retrieve relevant financial information\nQuery: {text}" for text in texts]

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                if self.is_e5_model:
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
                else:
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()

                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def _save_embeddings(self, embeddings_data: list, output_path: Path, append: bool = False):
        """Save embeddings to parquet file with safe append-only logic"""
        if not embeddings_data:
            return

        # Convert to DataFrame
        new_df = pd.DataFrame(embeddings_data)

        # Convert embeddings to columns
        embedding_matrix = np.vstack(new_df['embedding'].values)
        embedding_dim = embedding_matrix.shape[1]

        for i in range(embedding_dim):
            new_df[f'emb_{i:03d}'] = embedding_matrix[:, i]

        new_df = new_df.drop('embedding', axis=1)

        # Always use append mode to preserve existing embeddings
        if output_path.exists():
            logger.info(f"Loading existing embeddings from {output_path}")
            existing_df = pd.read_parquet(output_path)

            # Delete previous backups before creating new one
            from datetime import datetime
            backup_pattern = output_path.parent / "text_embeddings_backup_*.parquet"
            import glob
            old_backups = glob.glob(str(backup_pattern))
            for old_backup in old_backups:
                try:
                    Path(old_backup).unlink()
                    logger.info(f"Deleted old backup: {old_backup}")
                except Exception as e:
                    logger.warning(f"Could not delete old backup {old_backup}: {e}")

            # Create new backup
            backup_file = output_path.parent / f"text_embeddings_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            existing_df.to_parquet(backup_file, index=False)
            logger.info(f"Backup created: {backup_file}")
            
            # Identify truly new embeddings (not already in existing data)
            existing_keys = set(zip(existing_df['ticker'], existing_df['filing_date'].astype(str)))
            new_keys = set(zip(new_df['ticker'], new_df['filing_date'].astype(str)))
            
            # Only keep embeddings that are genuinely new
            truly_new_mask = new_df.apply(lambda x: (x['ticker'], str(x['filing_date'])) not in existing_keys, axis=1)
            truly_new_df = new_df[truly_new_mask]
            
            if len(truly_new_df) == 0:
                logger.info("No new embeddings to add - all embeddings already exist")
                return
            
            # Combine existing and new embeddings
            combined_df = pd.concat([existing_df, truly_new_df], ignore_index=True)
            
            # Final safety check: ensure we're not losing data
            if len(combined_df) < len(existing_df):
                logger.error("CRITICAL ERROR: Combined data has fewer rows than existing data!")
                logger.error("Restoring from backup to prevent data loss")
                existing_df.to_parquet(output_path, index=False)
                raise ValueError("Data integrity check failed - no changes made")
            
            combined_df.to_parquet(output_path, index=False)
            logger.info(f"Added {len(truly_new_df)} new embeddings (total: {len(combined_df)}, existing: {len(existing_df)})")
            
        else:
            # First time creating the file
            new_df.to_parquet(output_path, index=False)
            logger.info(f"Created new embeddings file with {len(new_df)} embeddings")

    def process_all_transcripts(self, input_file: str = "text_raw_comprehensive.parquet",
                                output_file: str = "text_embeddings.parquet",
                                batch_size: int = 16,
                                save_interval: int = 100,
                                num_workers: int = 4):
        """
        Process all transcripts and save embeddings with optimized batching

        Args:
            input_file: Input parquet file with text_content column
            output_file: Output parquet file for embeddings
            batch_size: Number of transcripts to process simultaneously (higher = better GPU utilization)
            save_interval: Save progress every N transcripts
            num_workers: Number of CPU workers for data loading
        """
        logger.info(f"Loading transcripts from {input_file}...")

        text_df = pd.read_parquet(self.data_dir / input_file)
        logger.info(f"Loaded {len(text_df)} transcripts")
        logger.info(f"Processing with batch size: {batch_size}, num_workers: {num_workers}")

        embeddings_data = []
        output_path = self.data_dir / output_file

        # Check if partial results exist and filter out already processed items
        if output_path.exists():
            logger.info(f"Found existing embeddings file, loading...")
            existing_df = pd.read_parquet(output_path)
            logger.info(f"Existing embeddings: {len(existing_df)} records")
            
            # Get already processed tickers/dates with more robust matching
            processed = set(zip(existing_df['ticker'], existing_df['filing_date'].astype(str)))
            initial_count = len(text_df)
            
            # Filter out already processed items
            text_df = text_df[~text_df.apply(lambda x: (x['ticker'], str(x['filing_date'])) in processed, axis=1)]
            
            logger.info(f"Filtered out {initial_count - len(text_df)} already processed transcripts")
            logger.info(f"Remaining to process: {len(text_df)} transcripts")
            
            # Additional safety check: warn if we're about to process items that might already exist
            if len(text_df) > 0:
                sample_keys = set(zip(text_df['ticker'].head(10), text_df['filing_date'].astype(str).head(10)))
                overlap = sample_keys.intersection(processed)
                if overlap:
                    logger.warning(f"WARNING: Found {len(overlap)} potential duplicates in sample - this should not happen with proper filtering")

        if len(text_df) == 0:
            logger.info("No transcripts to process")
            return pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()

        # Create dataset and dataloader
        dataset = TranscriptDataset(text_df, self.tokenizer, self.is_e5_model)
        collate_fn = partial(collate_chunks, tokenizer=self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,  # Disabled to reduce memory usage
            prefetch_factor=2 if num_workers > 0 else None
        )

        # Progress tracking
        start_time = time.time()
        processed_count = 0
        embedding_size = 4096 if self.is_e5_model else 768

        # Use tqdm for progress bar
        with tqdm(total=len(text_df), desc="Generating embeddings", unit="transcript") as pbar:
            for batch in dataloader:
                chunk_data = batch['chunk_data']
                tokenized = batch['tokenized']

                # Process all chunks in this batch at once
                if tokenized is not None:
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        tokenized = {k: v.to(self.device, non_blocking=True) for k, v in tokenized.items()}
                        outputs = self.model(**tokenized)

                        if self.is_e5_model:
                            chunk_embeddings = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
                        else:
                            chunk_embeddings = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()

                        # Clear GPU cache after each batch
                        del outputs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Aggregate chunks back to transcripts
                transcript_embeddings = {}
                chunk_idx = 0

                for item in chunk_data:
                    metadata = item['metadata']
                    transcript_idx = metadata['idx']

                    if item.get('is_empty', False):
                        # Empty transcript
                        embedding = np.zeros(embedding_size)
                    else:
                        # Get chunk embedding
                        embedding = chunk_embeddings[chunk_idx]
                        chunk_idx += 1

                    # Accumulate chunks for same transcript
                    if transcript_idx not in transcript_embeddings:
                        transcript_embeddings[transcript_idx] = {
                            'embeddings': [],
                            'metadata': metadata
                        }
                    transcript_embeddings[transcript_idx]['embeddings'].append(embedding)

                # Average chunks and save
                for transcript_idx, data in transcript_embeddings.items():
                    final_embedding = np.mean(data['embeddings'], axis=0)
                    metadata = data['metadata']

                    embeddings_data.append({
                        'ticker': metadata['ticker'],
                        'filing_date': metadata['filing_date'],
                        'document_type': metadata['document_type'],
                        'quarter': metadata['quarter'],
                        'year': metadata['year'],
                        'embedding': final_embedding
                    })

                    processed_count += 1

                # Update progress
                pbar.update(len(transcript_embeddings))
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                remaining = len(text_df) - processed_count
                eta_seconds = remaining / rate if rate > 0 else 0
                pbar.set_postfix({
                    'rate': f'{rate:.2f}/s',
                    'ETA': f'{eta_seconds/60:.1f}min'
                })

                # Save progress periodically
                if len(embeddings_data) >= save_interval:
                    self._save_embeddings(embeddings_data, output_path, append=True)  # Always append
                    embeddings_data = []

        # Save any remaining embeddings
        if embeddings_data:
            self._save_embeddings(embeddings_data, output_path, append=True)  # Always append

        # Load final result and verify integrity
        final_df = pd.read_parquet(output_path)
        embedding_cols = [col for col in final_df.columns if col.startswith('emb_')]
        embedding_dim = len(embedding_cols)

        # Verify file integrity
        integrity_check = self.verify_embeddings_integrity(output_path)
        
        logger.info("="*60)
        logger.info("FINANCIAL TEXT EMBEDDINGS COMPLETE")
        logger.info("="*60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Processed {len(final_df)} transcripts")
        logger.info(f"Embedding dimension: {embedding_dim}")
        logger.info(f"Saved to: {output_path}")
        logger.info(f"Output shape: {final_df.shape}")
        
        # Display integrity check results
        if integrity_check["status"] == "valid":
            logger.info("FILE INTEGRITY CHECK PASSED:")
            logger.info(f"  - Total records: {integrity_check['total_records']:,}")
            logger.info(f"  - Unique tickers: {integrity_check['unique_tickers']:,}")
            logger.info(f"  - Date range: {integrity_check['date_range'][0]} to {integrity_check['date_range'][1]}")
            logger.info(f"  - Embedding dimension: {integrity_check['embedding_dimension']}")
            logger.info(f"  - Duplicates: {integrity_check['duplicates']}")
            logger.info(f"  - Missing embeddings: {integrity_check['missing_embeddings']}")
            logger.info(f"  - File size: {integrity_check['file_size_mb']:.1f} MB")
        else:
            logger.warning(f"FILE INTEGRITY CHECK FAILED: {integrity_check['message']}")
            
        logger.info("="*60)

        return final_df

def main():
    """Run financial text embeddings generation"""
    try:
        embedder = FinBERTEmbeddings(data_dir="data_pipeline/data")
        embeddings_df = embedder.process_all_transcripts(
            input_file="text_raw_comprehensive.parquet",
            output_file="text_embeddings.parquet",
            batch_size=2,  # Increased batch size with 4-bit quantization
            save_interval=50,
            num_workers=1  # No multiprocessing to reduce memory overhead
        )

        print(f"\n[SUCCESS] Generated embeddings for {len(embeddings_df)} transcripts")
        print(f"[MODEL] {embedder.model_name}")
        print(f"[OUTPUT] data_pipeline/data/text_embeddings.parquet")

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

if __name__ == "__main__":
    main()
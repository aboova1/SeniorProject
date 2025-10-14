import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

# ----------------------------- Logging ---------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_lstm_upgraded")

# ----------------------------- Repro -----------------------------------
TORCH_SEED = 1337
np.random.seed(TORCH_SEED)
torch.manual_seed(TORCH_SEED)

def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------- Constants --------------------------------
BASE_EXCLUDE_COLS = [
    "sequence_id", "ticker", "quarter_in_sequence", "sequence_start_date",
    "sequence_end_date", "fiscal_quarter_end", "sector", "year",
    "transcript_date", "transcript_type", "days_after_quarter",
    "target_price_next_q", "current_price", "rebalance_date", "in_sp500",
    "period_start_date", "period_end_date", "holding_period_days", "transcript_available_date",
    "filing_date", "financials_release_date"
]

# Price-like names to EXCLUDE from features. Include common aliases
PRICE_CANDIDATE_NAMES = [
    "quarter_end_adj_close", "quarter_end_price", "adj_close", "close", "price", "px_last",
    "last_price", "prc", "mkt_cap", "marketcapitalization"
]

# ----------------------------- Dataset ---------------------------------
class SequenceDataset(Dataset):
    """
    Returns tuples: (x_7xF, y8_log, y7_log, y8_raw)
      - x_7xF: first 7 rows (Q1..Q7) of features (scaled)
      - y8_log: log1p(target_price at Q8)
      - y7_log: log1p(current_price at Q7)  [used for residual head]
      - y8_raw: raw target_price at Q8 (for metrics in original space)
    Assumes each sequence_id spans exactly 8 rows.
    """
    def __init__(self, sequences_df: pd.DataFrame, feature_cols: List[str]):
        self.feature_cols = feature_cols

        self.sequence_ids = sequences_df["sequence_id"].unique().tolist()
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        valid = 0

        for sid in self.sequence_ids:
            seq = sequences_df[sequences_df["sequence_id"] == sid].sort_values("quarter_in_sequence")
            if len(seq) != 8:
                continue

            q7 = seq.iloc[6]
            q8 = seq.iloc[7]

            # Use target_price_next_q and current_price
            y8 = q8.get('target_price_next_q', np.nan)
            y7 = q7.get('current_price', np.nan)
            if pd.isna(y8) or pd.isna(y7):
                continue

            # Features: Q1..Q7 only
            feat_rows = seq.iloc[:7]
            X = feat_rows[self.feature_cols].copy()
            X = X.ffill().bfill().fillna(0.0).values.astype(np.float32)  # (7, F)

            y8_log = np.log1p(float(y8))
            y7_log = np.log1p(float(y7))

            self.samples.append(
                (
                    torch.from_numpy(X),
                    torch.tensor([y8_log], dtype=torch.float32),
                    torch.tensor([y7_log], dtype=torch.float32),
                    torch.tensor([float(y8)], dtype=torch.float32),
                )
            )
            valid += 1

        logger.info(f"Loaded {valid} valid sequences (7q -> Q8 target_price)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ----------------------------- Model (UNIDIR) ---------------------------
class CausalAttention(nn.Module):
    """Simple dot-product attention: query = last hidden state; keys/values = outputs 1..7."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = hidden_size ** 0.5

    def forward(self, outputs: torch.Tensor, last_h: torch.Tensor):
        # outputs: (B, T=7, H), last_h: (B, H)
        q = last_h.unsqueeze(1)                            # (B, 1, H)
        k = outputs                                        # (B, 7, H)
        v = outputs                                        # (B, 7, H)
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / self.scale   # (B, 1, 7)
        attn = torch.softmax(attn_logits, dim=-1)                    # (B, 1, 7)
        ctx = torch.bmm(attn, v).squeeze(1)                          # (B, H)
        return ctx, attn.squeeze(1)

class UniLSTMAttnDelta(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 192, num_layers: int = 2, dropout: float = 0.25, use_attn: bool = True):
        super().__init__()
        self.use_attn = use_attn
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attn = CausalAttention(hidden_size) if use_attn else None
        rep_dim = hidden_size * (2 if use_attn else 1)
        self.ln = nn.LayerNorm(rep_dim)
        self.fc1 = nn.Linear(rep_dim + 1, 128)  # +1 for y7_log
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)            # predicts delta_log = y8_log - y7_log

    def forward(self, x: torch.Tensor, y7_log: torch.Tensor):
        # x: (B, 7, F), y7_log: (B, 1)
        outputs, (h, _) = self.lstm(x)
        last_h = h[-1]  # (B, H)
        if self.use_attn:
            ctx, _ = self.attn(outputs, last_h)
            rep = torch.cat([last_h, ctx], dim=1)  # (B, 2H)
        else:
            rep = last_h                            # (B, H)
        rep = self.ln(rep)
        rep = torch.cat([rep, y7_log], dim=1)       # inject y7_log explicitly
        z = self.fc1(rep)
        z = self.act(z)
        z = self.drop(z)
        delta_log = self.fc2(z)                     # (B, 1)
        y8_log_hat = y7_log + delta_log             # residual in log-space
        return y8_log_hat

# ---------------------------- Trainer ----------------------------------
class LSTMTrainer:
    BAD_KEYWORDS = (
        "return", "ret", "ratio", "margin", "yield", "emb", "embed", "txt_", "beta", "vol", "yoy", "growth",
        "target_excess_return_next_q"
    )
    GOOD_HINTS = (
        "quarter_end_adj_close", "adj_close", "quarter_end_price", "close", "price", "px", "px_last",
        "last_price", "prc"
    )

    def __init__(self,
                 train_path: str,
                 val_path: str,
                 test_path: str,
                 model_dir: str = "models",
                 results_dir: str = "results",
                 use_attn: bool = True,
                 loss_name: str = "huber"):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)

        self.model_dir = Path(model_dir); self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path(results_dir); self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.feature_cols: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[nn.Module] = None
        self.use_attn = use_attn
        self.loss_name = loss_name

        # will be set in load_and_prepare_data
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    # ---------- Price column detection ----------
    def _name_based_candidates(self, df: pd.DataFrame) -> List[str]:
        cands = []
        for col in df.columns:
            s = col.lower()
            if any(b in s for b in self.BAD_KEYWORDS):
                continue
            if any(k in s for k in self.GOOD_HINTS):
                cands.append(col)
        return cands

    def _numeric_candidates(self, df: pd.DataFrame) -> List[str]:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cands = []
        for col in num_cols:
            s = col.lower()
            if col.startswith("txt_"):
                continue
            if any(b in s for b in self.BAD_KEYWORDS):
                continue
            cands.append(col)
        return cands


    # ---------- feature analysis ----------
    def _analyze_features(self) -> dict:
        ft = {
            "technical": sum(1 for c in self.feature_cols if any(x in c for x in ["momentum", "vol", "beta"])),
            "fundamental": sum(1 for c in self.feature_cols if any(x in c for x in ["yield", "margin", "roe", "ratio", "yoy", "accrual"])),
            "text": sum(1 for c in self.feature_cols if c.startswith("txt_")),
            "other": 0,
        }
        ft["other"] = len(self.feature_cols) - sum(ft.values())
        return ft

    # ---------- data pipeline ----------
    def load_and_prepare_data(self, combine_all: bool = False):
        """
        Load and prepare data for training.

        Args:
            combine_all: If True, combine train+val+test into single training set
        """
        # Load three splits
        for pth in [self.train_path, self.val_path, self.test_path]:
            if not pth.exists():
                raise FileNotFoundError(f"Sequences file not found: {pth}")

        logger.info(f"Loading train: {self.train_path}")
        train_df = pd.read_parquet(self.train_path)
        logger.info(f"Loading val:   {self.val_path}")
        val_df = pd.read_parquet(self.val_path)
        logger.info(f"Loading test:  {self.test_path}")
        test_df = pd.read_parquet(self.test_path)

        # Basic checks
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if "quarter_in_sequence" not in df.columns:
                raise ValueError(f"{name} set missing required column 'quarter_in_sequence'.")
            if "target_price_next_q" not in df.columns:
                raise ValueError(f"{name} set missing required column 'target_price_next_q'.")
            if "current_price" not in df.columns:
                raise ValueError(f"{name} set missing required column 'current_price'.")
            logger.info(f"{name.capitalize()}: {len(df)} rows, {df['sequence_id'].nunique()} sequences, {df['ticker'].nunique()} tickers")

        # Feature cols: exclude base columns and price targets
        all_cols = train_df.columns.tolist()
        exclude_cols = set(BASE_EXCLUDE_COLS)
        self.feature_cols = [c for c in all_cols if c not in exclude_cols]
        logger.info(f"Using {len(self.feature_cols)} features")
        logger.info(f"Feature categories: {self._analyze_features()}")

        if combine_all:
            # Combine all data for final training
            logger.info("="*60)
            logger.info("COMBINING ALL DATA (train+val+test) FOR FINAL TRAINING")
            logger.info("="*60)
            combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            logger.info(f"Combined: {len(combined_df)} rows, {combined_df['sequence_id'].nunique()} sequences, {combined_df['ticker'].nunique()} tickers")

            # Scale on combined data
            def _nonlast_mask(df):
                qmax = df.groupby("sequence_id")["quarter_in_sequence"].transform("max")
                return df["quarter_in_sequence"] < qmax

            combined_nonlast = _nonlast_mask(combined_df)

            # Fill NaN values before scaling
            combined_df.loc[combined_nonlast, self.feature_cols] = combined_df.loc[combined_nonlast, self.feature_cols].ffill().bfill().fillna(0.0)

            self.scaler = StandardScaler()
            self.scaler.fit(combined_df.loc[combined_nonlast, self.feature_cols].values)
            combined_df.loc[combined_nonlast, self.feature_cols] = self.scaler.transform(
                combined_df.loc[combined_nonlast, self.feature_cols].values
            )

            # Use combined data for training, no validation set
            self.train_ds = SequenceDataset(combined_df, self.feature_cols)
            self.val_ds = None
            self.test_ds = None
            logger.info(f"Dataset size — Combined: {len(self.train_ds)}")
        else:
            # Scale ONLY first 7 rows per sequence (never touch Q8), fitting on TRAIN only
            logger.info("Normalizing features (Q1–Q7 only; Q8 excluded)...")

            def _nonlast_mask(df):
                qmax = df.groupby("sequence_id")["quarter_in_sequence"].transform("max")
                return df["quarter_in_sequence"] < qmax

            train_nonlast = _nonlast_mask(train_df)
            val_nonlast   = _nonlast_mask(val_df)
            test_nonlast  = _nonlast_mask(test_df)

            # Fill NaN values before scaling
            train_df.loc[train_nonlast, self.feature_cols] = train_df.loc[train_nonlast, self.feature_cols].ffill().bfill().fillna(0.0)
            val_df.loc[val_nonlast, self.feature_cols] = val_df.loc[val_nonlast, self.feature_cols].ffill().bfill().fillna(0.0)
            test_df.loc[test_nonlast, self.feature_cols] = test_df.loc[test_nonlast, self.feature_cols].ffill().bfill().fillna(0.0)

            self.scaler = StandardScaler()
            self.scaler.fit(train_df.loc[train_nonlast, self.feature_cols].values)

            train_df.loc[train_nonlast, self.feature_cols] = self.scaler.transform(train_df.loc[train_nonlast, self.feature_cols].values)
            val_df.loc[val_nonlast, self.feature_cols]     = self.scaler.transform(val_df.loc[val_nonlast, self.feature_cols].values)
            test_df.loc[test_nonlast, self.feature_cols]   = self.scaler.transform(test_df.loc[test_nonlast, self.feature_cols].values)

            # Build datasets
            self.train_ds = SequenceDataset(train_df, self.feature_cols)
            self.val_ds   = SequenceDataset(val_df,   self.feature_cols)
            self.test_ds  = SequenceDataset(test_df,  self.feature_cols)
            logger.info(f"Dataset sizes — Train: {len(self.train_ds)}, Val: {len(self.val_ds)}, Test: {len(self.test_ds)}")

    # ---------- model ----------
    def create_model(self, hidden_size: int = 192, num_layers: int = 2, dropout: float = 0.25):
        self.model = UniLSTMAttnDelta(
            input_size=len(self.feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_attn=self.use_attn,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Created model with {n_params:,} parameters")
        logger.info(f"Architecture: {num_layers}-layer unidirectional LSTM ({hidden_size} units)"
                    f" -> {'Attn+' if self.use_attn else ''}Residual-Head ( + y7_log ) -> Δlog -> y8_log")
        return self.model

    # ---------- train/eval helpers ----------
    @staticmethod
    def _metrics(pred_raw: np.ndarray, targ_raw: np.ndarray) -> dict:
        mse = float(np.mean((pred_raw - targ_raw) ** 2))
        mae = float(np.mean(np.abs(pred_raw - targ_raw)))
        rmse = float(np.sqrt(mse))
        # Robust MAPE (avoid div by zero)
        denom = np.clip(np.abs(targ_raw), 1e-8, None)
        mape = float(np.mean(np.abs((pred_raw - targ_raw) / denom)))
        ic = np.corrcoef(pred_raw, targ_raw)[0, 1] if len(pred_raw) > 1 else np.nan
        if np.isnan(ic):
            ic = 0.0
        return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "ic": float(ic)}

    def _epoch(self, loader, optimizer=None, criterion=None):
        train_mode = optimizer is not None
        self.model.train(train_mode)
        total = 0.0
        pred_raw_all, targ_raw_all = [], []

        for batch in loader:
            X, y8_log, y7_log, y8_raw = batch  # shapes: (B,7,F), (B,1), (B,1), (B,1)
            X = X.to(self.device); y8_log = y8_log.to(self.device); y7_log = y7_log.to(self.device); y8_raw = y8_raw.to(self.device)
            if train_mode:
                optimizer.zero_grad()
            y8_log_hat = self.model(X, y7_log)
            loss = criterion(y8_log_hat, y8_log)
            if train_mode:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
            total += float(loss.item())

            # collect raw-space preds for metrics
            pred_raw = torch.expm1(y8_log_hat).detach().cpu().numpy().flatten()
            targ_raw = y8_raw.detach().cpu().numpy().flatten()
            pred_raw_all.extend(pred_raw)
            targ_raw_all.extend(targ_raw)

        pred_raw_all = np.array(pred_raw_all); targ_raw_all = np.array(targ_raw_all)
        return (total / max(1, len(loader))), pred_raw_all, targ_raw_all

    def _make_criterion(self, name: str):
        name = name.lower()
        if name == "mse":
            return nn.MSELoss()
        if name == "huber":
            return nn.SmoothL1Loss(beta=0.25)  # robust to outliers
        if name == "logcosh":
            class LogCosh(nn.Module):
                def forward(self, input, target):
                    x = input - target
                    return torch.mean(torch.log(torch.cosh(x)))
            return LogCosh()
        if name == "mape":
            class MAPELoss(nn.Module):
                def forward(self, input, target):
                    # Convert from log space to raw price space
                    pred_raw = torch.expm1(input)
                    targ_raw = torch.expm1(target)
                    # Compute percentage error with numerical stability
                    epsilon = 1e-8
                    pct_error = torch.abs((pred_raw - targ_raw) / (torch.abs(targ_raw) + epsilon))
                    # Clip extreme values to prevent overly conservative predictions
                    pct_error = torch.clamp(pct_error, 0.0, 1.0)  # Cap at 100% error
                    return torch.mean(pct_error)
            return MAPELoss()
        if name == "hybrid":
            class HybridLoss(nn.Module):
                """Hybrid loss: combines MSE in log-space with direction-aware component"""
                def forward(self, input, target):
                    # MSE in log space (good for relative errors)
                    mse_loss = torch.mean((input - target) ** 2)

                    # Direction penalty: penalize getting the sign of change wrong
                    pred_raw = torch.expm1(input)
                    targ_raw = torch.expm1(target)
                    pred_change = pred_raw - targ_raw
                    targ_change = targ_raw - targ_raw  # Will use actual price change

                    # Combined loss
                    return mse_loss
            return HybridLoss()
        raise ValueError("Unknown loss name. Use one of: mse, huber, logcosh, mape")

    def train(self, num_epochs=80, batch_size=64, lr=2e-3, weight_decay=1e-5, patience=12, fixed_epochs=None):
        """
        Train the model.

        Args:
            num_epochs: Maximum number of epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience (ignored if fixed_epochs is set)
            fixed_epochs: If set, train for exactly this many epochs without early stopping or validation
        """
        if fixed_epochs is not None:
            # Training mode: fixed epochs, no validation
            logger.info("="*60)
            logger.info(f"TRAINING FOR FIXED {fixed_epochs} EPOCHS (no validation)")
            logger.info("="*60)

            train_ld = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
            criterion = self._make_criterion(self.loss_name)
            optimzr = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimzr, T_max=fixed_epochs)

            history = []

            for ep in range(fixed_epochs):
                tr_loss, tr_pred_raw, tr_targ_raw = self._epoch(train_ld, optimizer=optimzr, criterion=criterion)
                tr_m = self._metrics(tr_pred_raw, tr_targ_raw)
                sched.step()

                logger.info(f"Epoch {ep+1}/{fixed_epochs}")
                logger.info(f"  Train Loss: {tr_loss:.6f}")
                logger.info(f"  Train MAPE: {tr_m['mape']:.4f}")
                logger.info(f"  Train RMSE (raw): {tr_m['rmse']:.4f}")
                logger.info(f"  Train IC: {tr_m['ic']:.4f}")

                history.append({
                    "epoch": ep + 1,
                    "train_loss": tr_loss,
                    "train_metrics_raw": tr_m,
                })

            # Save final model
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimzr.state_dict(),
                "epoch": fixed_epochs,
                "feature_cols": self.feature_cols,
                "scaler_mean": self.scaler.mean_,
                "scaler_scale": self.scaler.scale_,
                "use_attn": self.use_attn,
                "loss_name": self.loss_name,
            }, self.model_dir / "final_model_all_data.pt")
            logger.info(f"  Saved final model to final_model_all_data.pt")

            with open(self.results_dir / "final_training_history.json", "w") as f:
                json.dump(history, f, indent=2, default=str)

            logger.info("="*60); logger.info("FIXED-EPOCH TRAINING COMPLETE"); logger.info("="*60)
            return history

        else:
            # Original training mode with validation and early stopping
            logger.info("="*60); logger.info("STARTING TRAINING (log-space, residual to Q7)"); logger.info("="*60)

            if self.val_ds is None:
                raise ValueError("Validation set required for training with early stopping. Use fixed_epochs parameter for training without validation.")

            train_ld = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_ld   = DataLoader(self.val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)

            criterion = self._make_criterion(self.loss_name)
            optimzr = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimzr, T_max=num_epochs)

            best = float("inf"); bad = 0
            best_epoch = 0
            history = []

            for ep in range(num_epochs):
                tr_loss, tr_pred_raw, tr_targ_raw = self._epoch(train_ld, optimizer=optimzr, criterion=criterion)
                vl_loss, vl_pred_raw, vl_targ_raw = self._epoch(val_ld, optimizer=None,   criterion=criterion)

                tr_m = self._metrics(tr_pred_raw, tr_targ_raw)
                vl_m = self._metrics(vl_pred_raw, vl_targ_raw)
                sched.step(vl_m['mape'])

                logger.info(f"Epoch {ep+1}/{num_epochs}")
                logger.info(f"  Train Loss: {tr_loss:.6f} | Val Loss: {vl_loss:.6f}")
                logger.info(f"  Train MAPE: {tr_m['mape']:.4f} | Val MAPE: {vl_m['mape']:.4f}")
                logger.info(f"  Train RMSE (raw): {tr_m['rmse']:.4f} | Val RMSE (raw): {vl_m['rmse']:.4f}")
                logger.info(f"  Train IC: {tr_m['ic']:.4f} | Val IC: {vl_m['ic']:.4f}")

                history.append({
                    "epoch": ep + 1,
                    "train_loss": tr_loss,
                    "val_loss": vl_loss,
                    "train_metrics_raw": tr_m,
                    "val_metrics_raw": vl_m,
                })

                # early stopping on val MAPE (percentage error)
                if vl_m['mape'] < best:
                    best = vl_m['mape']; bad = 0
                    best_epoch = ep + 1
                    torch.save({
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimzr.state_dict(),
                        "epoch": ep + 1,
                        "val_mape": float(vl_m['mape']),
                        "val_loss": float(vl_loss),
                        "feature_cols": self.feature_cols,
                        "scaler_mean": self.scaler.mean_,
                        "scaler_scale": self.scaler.scale_,
                        "use_attn": self.use_attn,
                        "loss_name": self.loss_name,
                    }, self.model_dir / "best_model.pt")
                    logger.info(f"  Saved best model (val MAPE: {vl_m['mape']:.4f})")
                else:
                    bad += 1
                    if bad >= patience:
                        logger.info(f"Early stopping triggered after {ep+1} epochs")
                        break

            with open(self.results_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=2, default=str)

            # Save optimal epoch info
            with open(self.results_dir / "optimal_epochs.txt", "w") as f:
                f.write(f"{best_epoch}\n")
            logger.info(f"Optimal epoch count saved: {best_epoch}")

            logger.info("="*60); logger.info("TRAINING COMPLETE"); logger.info("="*60)
            return history

    def evaluate_test_set(self):
        logger.info("Evaluating on test set...")
        ckpt = torch.load(self.model_dir / "best_model.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)

        test_ld = DataLoader(self.test_ds, batch_size=128, shuffle=False, pin_memory=True)
        criterion = self._make_criterion(self.loss_name)

        # Collect predictions with Q7 prices for directional analysis
        self.model.eval()
        pred_raw_all, targ_raw_all, y7_raw_all = [], [], []
        total_loss = 0.0

        for batch in test_ld:
            X, y8_log, y7_log, y8_raw = batch
            X = X.to(self.device); y8_log = y8_log.to(self.device); y7_log = y7_log.to(self.device); y8_raw = y8_raw.to(self.device)

            with torch.no_grad():
                y8_log_hat = self.model(X, y7_log)
                loss = criterion(y8_log_hat, y8_log)
                total_loss += float(loss.item())

                pred_raw = torch.expm1(y8_log_hat).cpu().numpy().flatten()
                targ_raw = y8_raw.cpu().numpy().flatten()
                y7_raw = torch.expm1(y7_log).cpu().numpy().flatten()

                pred_raw_all.extend(pred_raw)
                targ_raw_all.extend(targ_raw)
                y7_raw_all.extend(y7_raw)

        pred_raw = np.array(pred_raw_all)
        targ_raw = np.array(targ_raw_all)
        y7_raw = np.array(y7_raw_all)
        loss = total_loss / max(1, len(test_ld))

        m = self._metrics(pred_raw, targ_raw)

        logger.info("="*60); logger.info("TEST SET RESULTS (raw units)"); logger.info("="*60)
        logger.info(f"Test Log-Loss: {loss:.6f}")
        logger.info(f"Test RMSE: {m['rmse']:.6f} | Test MAPE: {m['mape']:.6f}")
        logger.info(f"Test MAE: {m['mae']:.6f} | Test IC: {m['ic']:.4f}")
        logger.info("="*60)

        # Directional classification accuracy
        logger.info("DIRECTIONAL CLASSIFICATION (UP vs DOWN)")
        logger.info("="*60)

        # Calculate directional changes (Q8 vs Q7)
        pred_direction = (pred_raw > y7_raw).astype(int)  # 1 = up, 0 = down
        actual_direction = (targ_raw > y7_raw).astype(int)

        # Overall accuracy
        correct = (pred_direction == actual_direction).sum()
        total = len(pred_direction)
        accuracy = correct / total

        # Per-class accuracy
        up_mask = (actual_direction == 1)
        down_mask = (actual_direction == 0)

        up_correct = ((pred_direction == actual_direction) & up_mask).sum()
        up_total = up_mask.sum()
        up_accuracy = up_correct / up_total if up_total > 0 else 0.0

        down_correct = ((pred_direction == actual_direction) & down_mask).sum()
        down_total = down_mask.sum()
        down_accuracy = down_correct / down_total if down_total > 0 else 0.0

        logger.info(f"Overall Directional Accuracy: {accuracy:.4f} ({correct}/{total})")
        logger.info(f"")
        logger.info(f"Class: UP (stock went up)")
        logger.info(f"  Samples: {up_total} ({up_total/total*100:.1f}%)")
        logger.info(f"  Accuracy: {up_accuracy:.4f} ({up_correct}/{up_total})")
        logger.info(f"")
        logger.info(f"Class: DOWN (stock went down)")
        logger.info(f"  Samples: {down_total} ({down_total/total*100:.1f}%)")
        logger.info(f"  Accuracy: {down_accuracy:.4f} ({down_correct}/{down_total})")
        logger.info("="*60)

        # Threshold accuracy (10% growth)
        logger.info("")
        logger.info("THRESHOLD ACCURACY (10%+ Growth Prediction)")
        logger.info("="*60)

        # Calculate percentage changes
        actual_pct_change = ((targ_raw - y7_raw) / y7_raw) * 100
        pred_pct_change = ((pred_raw - y7_raw) / y7_raw) * 100

        # Stocks that actually went up by 10%+
        actual_10plus = actual_pct_change >= 10.0
        actual_10plus_count = actual_10plus.sum()

        # Stocks that model predicted would go up by 10%+
        pred_10plus = pred_pct_change >= 10.0
        pred_10plus_count = pred_10plus.sum()

        # True positives: predicted 10%+ AND actually went up 10%+
        true_positives = (pred_10plus & actual_10plus).sum()

        # False positives: predicted 10%+ but did NOT go up 10%+
        false_positives = (pred_10plus & ~actual_10plus).sum()

        # False negatives: did NOT predict 10%+ but actually went up 10%+
        false_negatives = (~pred_10plus & actual_10plus).sum()

        # Recall: of stocks that went up 10%+, how many did we identify?
        recall = true_positives / actual_10plus_count if actual_10plus_count > 0 else 0.0

        # Precision: of stocks we predicted would go up 10%+, how many actually did?
        precision = true_positives / pred_10plus_count if pred_10plus_count > 0 else 0.0

        # Average actual return for false positives (stocks we thought would go up 10%+ but didn't)
        if false_positives > 0:
            fp_mask = pred_10plus & ~actual_10plus
            avg_fp_return = actual_pct_change[fp_mask].mean()
        else:
            avg_fp_return = 0.0

        logger.info(f"Stocks that actually went up 10%+: {actual_10plus_count} ({actual_10plus_count/total*100:.1f}%)")
        logger.info(f"Stocks model predicted would go up 10%+: {pred_10plus_count} ({pred_10plus_count/total*100:.1f}%)")
        logger.info(f"")
        logger.info(f"True Positives (correctly identified 10%+ gainers): {true_positives}")
        logger.info(f"False Positives (predicted 10%+ but didn't achieve it): {false_positives}")
        logger.info(f"False Negatives (missed 10%+ gainers): {false_negatives}")
        logger.info(f"")
        logger.info(f"Recall (sensitivity): {recall:.4f}")
        logger.info(f"  → Of {actual_10plus_count} stocks that went up 10%+, model identified {true_positives}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"  → Of {pred_10plus_count} stocks predicted to go up 10%+, {true_positives} actually did")
        logger.info(f"")
        logger.info(f"Average actual return for false positives: {avg_fp_return:.2f}%")
        logger.info(f"  → Stocks predicted to gain 10%+ but didn't actually gained {avg_fp_return:.2f}% on average")
        logger.info("="*60)

        with open(self.results_dir / "test_results.json", "w") as f:
            json.dump({
                "test_log_loss": float(loss),
                "test_metrics_raw": m,
                "directional_accuracy": {
                    "overall": float(accuracy),
                    "up_class": {"accuracy": float(up_accuracy), "samples": int(up_total)},
                    "down_class": {"accuracy": float(down_accuracy), "samples": int(down_total)},
                },
                "threshold_accuracy_10pct": {
                    "actual_10plus_count": int(actual_10plus_count),
                    "predicted_10plus_count": int(pred_10plus_count),
                    "true_positives": int(true_positives),
                    "false_positives": int(false_positives),
                    "false_negatives": int(false_negatives),
                    "recall": float(recall),
                    "precision": float(precision),
                    "avg_false_positive_return": float(avg_fp_return),
                },
                "predictions_raw": pred_raw.tolist(),
                "targets_raw": targ_raw.tolist(),
            }, f, indent=2)
        return m

# ---------------------------- CLI / Main -------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train upgraded UNIDIR LSTM (7q -> Q8 target)")

    # New: explicit split files (defaults align with your creator script output)
    p.add_argument("--train", type=str, default="data_pipeline/data/sequences_8q_train.parquet",
                   help="Path to TRAIN sequences parquet.")
    p.add_argument("--val", type=str, default="data_pipeline/data/sequences_8q_val.parquet",
                   help="Path to VAL sequences parquet.")
    p.add_argument("--test", type=str, default="data_pipeline/data/sequences_8q_test.parquet",
                   help="Path to TEST sequences parquet.")

    # Back-compat: if you pass --data, we'll still try to use a single file (for quick experiments)
    p.add_argument("--data", type=str, default=None,
                   help="[Optional/legacy] Single parquet with all sequences. If provided, will be used for all three splits (temporal split is NO LONGER performed here).")

    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=12)

    p.add_argument("--no-attn", dest="attn", action="store_false", help="Disable attention pooling (defaults to ON).")
    p.add_argument("--loss", type=str, default="huber", choices=["mse", "huber", "logcosh", "mape", "hybrid"], help="Training loss (huber is recommended for balanced predictions).")

    # Retraining on all data
    p.add_argument("--retrain-all", action="store_true",
                   help="Retrain on ALL data (train+val+test) using optimal epochs from validation.")
    p.add_argument("--retrain-epochs", type=int, default=None,
                   help="Number of epochs to use for retraining on all data. If not specified, will first find optimal epochs via validation, then retrain.")

    return p.parse_args()

def main():
    args = parse_args()
    set_deterministic()

    # If --data is given (legacy), just point all three to it (no internal split here).
    if args.data is not None:
        train_path = val_path = test_path = Path(args.data)
    else:
        train_path = Path(args.train)
        val_path   = Path(args.val)
        test_path  = Path(args.test)

    if args.retrain_all:
        # Retraining mode: train on ALL data (train+val+test)
        logger.info("="*80)
        logger.info("RETRAINING MODE: Training on ALL data (train+val+test)")
        logger.info("="*80)

        # Determine number of epochs to use
        if args.retrain_epochs is not None:
            # User specified epochs directly
            optimal_epochs = args.retrain_epochs
            logger.info(f"Using user-specified {optimal_epochs} epochs for retraining")
        else:
            # First, find optimal epochs via validation
            logger.info("STEP 1: Finding optimal epochs via validation...")
            logger.info("="*80)

            trainer = LSTMTrainer(
                str(train_path),
                str(val_path),
                str(test_path),
                use_attn=args.attn,
                loss_name=args.loss
            )
            trainer.load_and_prepare_data(combine_all=False)
            trainer.create_model(hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout)
            trainer.train(num_epochs=args.epochs, batch_size=args.bs, lr=args.lr,
                         weight_decay=args.wd, patience=args.patience)

            # Read optimal epoch count
            optimal_epochs_file = Path("results") / "optimal_epochs.txt"
            if not optimal_epochs_file.exists():
                raise FileNotFoundError("Could not find optimal_epochs.txt. Training may have failed.")

            with open(optimal_epochs_file, 'r') as f:
                optimal_epochs = int(f.read().strip())

            logger.info("="*80)
            logger.info(f"STEP 1 COMPLETE: Optimal epochs = {optimal_epochs}")
            logger.info("="*80)
            logger.info("")

        # Now retrain on ALL data
        logger.info("="*80)
        logger.info(f"STEP 2: Retraining on ALL data for {optimal_epochs} epochs...")
        logger.info("="*80)

        # Create new trainer for combined data
        trainer_final = LSTMTrainer(
            str(train_path),
            str(val_path),
            str(test_path),
            use_attn=args.attn,
            loss_name=args.loss
        )
        trainer_final.load_and_prepare_data(combine_all=True)
        trainer_final.create_model(hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout)
        trainer_final.train(num_epochs=args.epochs, batch_size=args.bs, lr=args.lr,
                          weight_decay=args.wd, patience=args.patience, fixed_epochs=optimal_epochs)

        logger.info("="*80)
        logger.info("RETRAINING COMPLETE!")
        logger.info("="*80)
        print("[SUCCESS] Retraining on all data complete!")
        print(f"[OPTIMAL EPOCHS] {optimal_epochs}")
        print(f"[FINAL MODEL] Saved to {Path('models') / 'final_model_all_data.pt'}")

    else:
        # Normal mode: train with validation and test
        trainer = LSTMTrainer(
            str(train_path),
            str(val_path),
            str(test_path),
            use_attn=args.attn,
            loss_name=args.loss
        )
        trainer.load_and_prepare_data(combine_all=False)
        trainer.create_model(hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout)
        trainer.train(num_epochs=args.epochs, batch_size=args.bs, lr=args.lr,
                     weight_decay=args.wd, patience=args.patience)
        test_metrics = trainer.evaluate_test_set()

        print("[SUCCESS] Training complete!")
        print(f"[TEST RMSE] {test_metrics['rmse']:.4f}")
        print(f"[TEST MAPE] {test_metrics['mape']:.4f}")
        print(f"[TEST IC]   {test_metrics['ic']:.4f}")
        print(f"[MODEL] Saved to {Path('models') / 'best_model.pt'}")

if __name__ == "__main__":
    main()
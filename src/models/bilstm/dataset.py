from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger


# ── 13 features per timestep ───────────────────────────────────────────────────
SEQ_FEATURES = [
    # ── Raw temporal / velocity ──────────────────────────────────────────────
    "TransactionAmt_log",
    "amt_zscore",
    "uid_gap_last",
    "hour_sin",
    "hour_cos",
    "is_late_night",
    "uid_tx_1h",
    "uid_total_amt_24h",
    "D1_norm",
    "device_mobile",
    "ProductCD_encoded",
    "M_match_count",
    # ── Meta-feature: stacking ensemble output (OOF in train, live in prod) ─
    # "calibrated_stacking_score",
]

SEQ_LEN = 10   # 95-th pct of train users ≤ 7 txns → 10 captures the bulk cleanly


# ── Cohort prior cache (loaded lazily) ────────────────────────────────────────
_COHORT_PRIORS: Optional[dict] = None
_COHORT_PRIORS_PATH = Path("data/processed/cohort_priors.json")

COHORT_COLS = ["DeviceType", "P_emaildomain", "ProductCD"]


def _load_cohort_priors() -> dict:
    """Lazily load cohort priors JSON (built by build_cohort_priors.py)."""
    global _COHORT_PRIORS
    if _COHORT_PRIORS is None:
        if _COHORT_PRIORS_PATH.exists():
            with open(_COHORT_PRIORS_PATH) as f:
                _COHORT_PRIORS = json.load(f)
            logger.info("Loaded {} cohort priors from {}", len(_COHORT_PRIORS), _COHORT_PRIORS_PATH)
        else:
            logger.warning(
                "Cohort priors not found at {}. Using zero padding (run build_cohort_priors.py first).",
                _COHORT_PRIORS_PATH,
            )
            _COHORT_PRIORS = {}
    return _COHORT_PRIORS


def _make_cohort_key(row: pd.Series) -> str:
    """Build composite cohort key matching build_cohort_priors.py logic."""
    parts = []
    for col in COHORT_COLS:
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            parts.append("unknown")
        else:
            parts.append(str(val).strip().lower())
    return "_".join(parts)


def _get_pad_vector(cohort_key: str, priors: dict, feature_cols: list[str]) -> np.ndarray:
    """
    Return a 1-D float32 vector of cohort-average values for padding.
    Falls back to '__global__' then zeros if no priors available.
    """
    stats = priors.get(cohort_key) or priors.get("__global__")
    if stats is None:
        return np.zeros(len(feature_cols), dtype=np.float32)

    return np.array(
        [stats.get(f"{col}_mean", 0.0) for col in feature_cols],
        dtype=np.float32,
    )


def build_sequences(
    df: pd.DataFrame,
    uid_col: str = "uid1",
    seq_len: int = SEQ_LEN,
    feature_cols: List[str] = SEQ_FEATURES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (N, seq_len, n_features) array by looking back at each uid's history.

    Train/serve skew fixes applied here:
      - Sorts by [TransactionDT, TransactionID] before sequence construction
      - Pads with COHORT-AVERAGE values (not zeros)

    For transaction i with uid u:
      - Collect the previous (seq_len-1) transactions by uid u (before i)
      - Append transaction i as the last step
      - Left-pad with cohort-average phantom steps if fewer than seq_len-1 prior txns

    Returns
    -------
    sequences : float32 array of shape (N, seq_len, n_features)
    labels    : int array of shape (N,)
    """
    # ── Sort guarantee (skew fix #4) ─────────────────────────────────────────
    sort_cols = [c for c in ["TransactionDT", "TransactionID"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        logger.warning("TransactionDT/TransactionID missing — sequence order may differ from Redis!")

    present_features = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("BiLSTM missing features (will be zeroed): {}", missing)

    n      = len(df)
    n_feat = len(feature_cols)

    # ── Load cohort priors for padding (skew fix #1) ─────────────────────────
    priors = _load_cohort_priors()
    use_cohort_padding = bool(priors)

    # Pre-compute cohort key per row (needed for padding)
    if use_cohort_padding:
        cohort_keys = df.apply(_make_cohort_key, axis=1).values
    else:
        cohort_keys = np.array([""] * n)

    # ── Feature matrix ───────────────────────────────────────────────────────
    feat_arr = np.zeros((n, n_feat), dtype=np.float32)
    for j, col in enumerate(feature_cols):
        if col in df.columns:
            feat_arr[:, j] = df[col].fillna(0).values.astype(np.float32)

    # ── Per-uid history index ─────────────────────────────────────────────────
    uid_vals        = df[uid_col].values
    uid_to_indices: dict = {}
    uid_to_pos:     dict = {}
    for i, u in enumerate(uid_vals):
        if u not in uid_to_indices:
            uid_to_indices[u] = []
        uid_to_pos[i] = len(uid_to_indices[u])
        uid_to_indices[u].append(i)

    logger.info(
        "Building BiLSTM sequences: N={}, seq_len={}, features={} (cohort_padding={})…",
        n, seq_len, n_feat, use_cohort_padding,
    )

    sequences = np.zeros((n, seq_len, n_feat), dtype=np.float32)

    for i in range(n):
        uid   = uid_vals[i]
        hist  = uid_to_indices[uid]
        pos   = uid_to_pos[i]
        prior = hist[max(0, pos - (seq_len - 1)) : pos]

        seq_indices = prior + [i]
        seq_steps   = feat_arr[seq_indices]           # shape (k+1, n_feat)

        pad_len = seq_len - len(seq_steps)
        if pad_len > 0:
            if use_cohort_padding:
                pad_vec = _get_pad_vector(cohort_keys[i], priors, feature_cols)
                pad_block = np.tile(pad_vec, (pad_len, 1))    # (pad_len, n_feat)
            else:
                pad_block = np.zeros((pad_len, n_feat), dtype=np.float32)

            seq_steps = np.vstack([pad_block, seq_steps])

        sequences[i] = seq_steps[-seq_len:]   # safety clip

    labels = (
        df["isFraud"].values.astype(np.int64)
        if "isFraud" in df.columns
        else np.zeros(n, dtype=np.int64)
    )

    logger.info("BiLSTM sequences built: shape={}", sequences.shape)
    return sequences, labels


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class BiLSTMDataset(Dataset):
    """PyTorch Dataset wrapping pre-built sequence arrays."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray = None):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

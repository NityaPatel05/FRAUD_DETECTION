from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# ── allow running as script from project root ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.bilstm.dataset import SEQ_FEATURES

# Columns used to build the composite cohort key
COHORT_COLS = ["DeviceType", "P_emaildomain", "ProductCD"]

# Fallback value for unseen categories
_FALLBACK = "unknown"


def _safe_str(val) -> str:
    """Normalise a raw cell to a lowercase string, replacing NaN with 'unknown'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return _FALLBACK
    return str(val).strip().lower()


def make_cohort_key(row: pd.Series) -> str:
    """
    Build a cohort identifier from the three defining columns.
    Example: 'mobile_gmail.com_w'
    """
    parts = [_safe_str(row.get(c, _FALLBACK)) for c in COHORT_COLS]
    return "_".join(parts)


def compute_cohort_priors(
    train_df: pd.DataFrame,
    feature_cols: list[str] = SEQ_FEATURES,
) -> dict:
    """
    Compute per-cohort means, stds, and fraud rate from the training set.

    Parameters
    ----------
    train_df    : Training DataFrame (MUST contain 'isFraud' column).
    feature_cols: List of BiLSTM seq features to aggregate.

    Returns
    -------
    priors : dict  {cohort_key: {feat_mean, feat_std, fraud_rate, count}}
    """
    # Build composite key column
    df = train_df.copy()
    df["_cohort_key"] = df.apply(make_cohort_key, axis=1)

    present_feats = [c for c in feature_cols if c in df.columns]
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        logger.warning("Cohort priors: features not in df (will be 0): {}", missing_feats)

    # Per-cohort aggregation
    group = df.groupby("_cohort_key")

    priors: dict = {}

    # Global fallback stats (for cohorts not seen at runtime)
    global_means = {c: float(df[c].mean()) if c in df.columns else 0.0 for c in feature_cols}
    global_stds  = {c: float(df[c].std())  if c in df.columns else 1.0 for c in feature_cols}
    global_fraud_rate = float(df["isFraud"].mean()) if "isFraud" in df.columns else 0.035

    for cohort_key, g in group:
        stats: dict = {
            "count":      int(len(g)),
            "fraud_rate": float(g["isFraud"].mean()) if "isFraud" in g.columns else global_fraud_rate,
        }
        for col in feature_cols:
            if col in g.columns:
                vals = g[col].dropna()
                stats[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else global_means[col]
                stats[f"{col}_std"]  = float(vals.std())  if len(vals) > 1 else global_stds[col]
            else:
                stats[f"{col}_mean"] = 0.0
                stats[f"{col}_std"]  = 1.0

        priors[cohort_key] = stats

    # Always include a global-average fallback cohort
    global_stats: dict = {
        "count":      int(len(df)),
        "fraud_rate": global_fraud_rate,
    }
    for col in feature_cols:
        global_stats[f"{col}_mean"] = global_means[col]
        global_stats[f"{col}_std"]  = global_stds[col]
    priors["__global__"] = global_stats

    logger.info(
        "Cohort priors computed: {} cohorts (+ 1 global fallback). "
        "Global fraud rate: {:.3%}",
        len(priors) - 1, global_fraud_rate,
    )
    return priors


def save_cohort_priors(priors: dict, out_paths: list[str]) -> None:
    """Serialise priors dict to one or more JSON files."""
    for path_str in out_paths:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(priors, f, indent=2)
        logger.info("Cohort priors saved → {}", p)


def load_cohort_priors(path: str) -> dict:
    """Load priors from a JSON file."""
    with open(path) as f:
        return json.load(f)


def get_cohort_stats(
    priors: dict,
    cohort_key: str,
    feature_cols: list[str] = SEQ_FEATURES,
) -> dict:
    """
    Return stats for a given cohort key.
    Falls back to '__global__' if the exact cohort has never been seen.
    """
    if cohort_key in priors:
        return priors[cohort_key]
    logger.debug("Cohort '{}' not found — using global fallback.", cohort_key)
    return priors["__global__"]


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    cfg_path = Path("configs/config.yaml")
    if not cfg_path.exists():
        logger.error("configs/config.yaml not found. Run from project root.")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    bilstm_dir = Path(cfg["paths"]["processed_bilstm"])
    art_dir    = Path(cfg["paths"]["checkpoints"])
    proc_dir   = Path("data/processed")

    train_path = bilstm_dir / "train.parquet"
    if not train_path.exists():
        logger.error("BiLSTM train.parquet not found at {}. Run preprocessing first.", train_path)
        sys.exit(1)

    logger.info("Loading BiLSTM train dataset from {} …", train_path)
    train_df = pd.read_parquet(train_path)
    logger.info("Loaded {} rows, {} columns.", len(train_df), train_df.shape[1])

    priors = compute_cohort_priors(train_df)

    save_cohort_priors(priors, [
        str(proc_dir / "cohort_priors.json"),
        str(art_dir  / "cohort_priors.json"),
    ])

    # Quick summary
    top5 = sorted(priors.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
    logger.info("Top 5 cohorts by size:")
    for k, v in top5:
        logger.info("  {:40s}  n={:7d}  fraud_rate={:.3%}", k, v["count"], v["fraud_rate"])


if __name__ == "__main__":
    main()

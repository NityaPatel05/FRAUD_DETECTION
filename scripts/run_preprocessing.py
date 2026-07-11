import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from pathlib import Path

import yaml
from loguru import logger

from src.preprocessing.sort_and_split import load_raw_data, sort_by_time, time_split
from src.preprocessing.v_column_reduction import group_v_cols_by_nan, reduce_v_by_correlation
from src.utils.adversarial_validation import run_adversarial_validation


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    raw_dir    = cfg["paths"]["raw_dir"]
    out_dir    = Path("data/processed")
    art_dir    = Path(cfg["paths"]["checkpoints"])
    art_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Preprocessing Pipeline ===")

    # ── Load and sort ─────────────────────────────────────────────────────────
    logger.info("Loading raw data from {}…", raw_dir)
    train_df, test_df = load_raw_data(raw_dir)

    train_df = sort_by_time(train_df)
    test_df  = sort_by_time(test_df)

    # ── Time split ────────────────────────────────────────────────────────────
    train_split, val_split = time_split(train_df, cfg["split"]["train_quantile"])

    # ── V column groups (fit on train only) ───────────────────────────────────
    corr_thr  = cfg["feature_engineering"]["v_col_corr_threshold"]
    v_groups  = group_v_cols_by_nan(train_split)
    kept_v    = reduce_v_by_correlation(train_split, v_groups, corr_threshold=corr_thr)

    with open(art_dir / "v_groups.pkl", "wb") as f:
        pickle.dump({"groups": v_groups, "kept_v_cols": kept_v}, f)
    logger.info("V column reduction: {} → {} columns saved.", sum(len(g) for g in v_groups.values()), len(kept_v))

    # ── Adversarial validation ────────────────────────────────────────────────
    numeric_cols = [c for c in train_split.columns if train_split[c].dtype != object and c != "isFraud"]
    safe_cols, adv_report = run_adversarial_validation(
        train_split, test_df,
        feature_cols=numeric_cols[:100],   # quick pass on first 100 features
        threshold=cfg["adversarial"]["drop_threshold"],
    )
    adv_report.to_csv(art_dir / "adversarial_validation_report.csv", index=False)
    logger.info("Adversarial validation done. Report saved.")

    # ── Save splits ───────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    train_split.to_parquet(out_dir / "train_raw.parquet", index=False)
    val_split.to_parquet(out_dir / "val_raw.parquet",     index=False)
    test_df.to_parquet(out_dir / "test_raw.parquet",      index=False)
    logger.info("Raw sorted splits saved to {}", out_dir)
    logger.info("=== Preprocessing Complete ===")


if __name__ == "__main__":
    main()

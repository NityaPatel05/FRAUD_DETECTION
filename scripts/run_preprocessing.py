"""
Step 1 — load raw CSVs, sort by time, make ONE time split, save raw splits.

  train = labelled rows with day <= quantile(train_quantile)   (≈ days 1–130)
  val   = remaining labelled rows                              (≈ days 131–182)
  test  = Kaggle test (unlabelled)

Removed on purpose (see audit):
  * the adversarial "drift" filter against Kaggle test — it deleted the most
    predictive columns (TransactionAmt, card1-6, C, D …). Feature selection now
    uses a time-consistency check instead (scripts/run_feature_selection.py).
  * V-column reduction here — moved into the v2 feature pipeline.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import yaml
from loguru import logger

from src.preprocessing.sort_and_split import load_raw_data, sort_by_time, time_split


def main():
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Preprocessing ===")
    train_df, test_df = load_raw_data(cfg["paths"]["raw_dir"])   # renames test id-xx → id_xx
    train_df = sort_by_time(train_df)
    test_df = sort_by_time(test_df)

    train_split, val_split = time_split(train_df, cfg["split"]["train_quantile"])

    train_split.to_parquet(out_dir / "train_raw.parquet", index=False)
    val_split.to_parquet(out_dir / "val_raw.parquet", index=False)
    test_df.to_parquet(out_dir / "test_raw.parquet", index=False)
    logger.info("Saved raw splits to {} (train {}, val {}, test {})",
                out_dir, len(train_split), len(val_split), len(test_df))


if __name__ == "__main__":
    main()

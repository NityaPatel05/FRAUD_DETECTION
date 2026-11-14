"""
Step 3 — time-consistency feature selection (train period only; val untouched).

  python scripts/run_feature_selection.py [--agg-mode causal|transductive] [--threshold 0.5]

Writes artifacts/v2/<mode>/time_consistency_report.csv and dropped_features.json,
which scripts/run_training.py applies automatically (use --no-selection to ignore).
Always confirm with run_training.py that the drop does not reduce CV AUC-PR.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.utils.time_consistency import time_consistency_check


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    mode = args.agg_mode or cfg["fe_v2"]["agg_mode"]
    from src.feature_engineering.pipeline_v2 import dataset_tag
    tag = dataset_tag(mode, args.schema or cfg["fe_v2"].get("schema", "full"))
    feat_dir = Path(cfg["paths"]["processed_v2"].format(mode=tag))
    out_dir = Path(cfg["paths"]["artifacts_v2"].format(mode=tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((feat_dir / "meta.json").read_text())
    train = pd.read_parquet(feat_dir / "train.parquet")
    rep = time_consistency_check(train, meta["feature_cols"], args.window_days, args.threshold)
    if args.two_windows:
        # data study: days 1-29 are a different (December) regime, so a feature is dropped
        # only if it ALSO fails when trained on the post-December period
        start = int(train["day"].min()) + 30
        rep2 = time_consistency_check(train, meta["feature_cols"], args.window_days, args.threshold,
                                      train_start_day=start, train_end_day=int(train["day"].max()) - args.window_days)
        rep = rep.merge(rep2[["feature", "auc_first_month", "auc_last_month", "drop"]], on="feature",
                        suffixes=("", "_window2"))
        rep["drop"] = rep["drop"] & rep["drop_window2"]
    rep.to_csv(out_dir / "time_consistency_report.csv", index=False)
    dropped = rep.loc[rep["drop"], "feature"].tolist()
    (out_dir / "dropped_features.json").write_text(json.dumps({"threshold": args.threshold, "dropped": dropped}, indent=1))
    print(f"{len(dropped)} features flagged: {dropped}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agg-mode", choices=["causal", "transductive"], default=None)
    p.add_argument("--schema", choices=["full", "raw"], default=None)
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--two-windows", action="store_true",
                   help="also test train=day 31..(end-30) -> last month; drop only if both fail")
    main(p.parse_args())

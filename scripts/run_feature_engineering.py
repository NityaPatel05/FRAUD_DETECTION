"""
Step 2 — build v2 tree-model features.

Usage:
  python scripts/run_feature_engineering.py                       # agg_mode from config (causal)
  python scripts/run_feature_engineering.py --agg-mode transductive
  python scripts/run_feature_engineering.py --schema raw          # real-world fields only

Reads   data/processed/{train,val,test}_raw.parquet  (from run_preprocessing.py)
Writes  data/processed/v2_<mode>/{train,val,test}.parquet + meta.json

The splits are passed in as-is: there is NO second time split here (the old
pipeline split train again, trained on days 1–95 and "validated" on 96–130,
while the real validation window ended up inside test).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from src.feature_engineering.pipeline_v2 import run_feature_pipeline_v2, dataset_tag


def main(args):
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    fe_cfg = dict(cfg.get("fe_v2", {}))
    if args.agg_mode:
        fe_cfg["agg_mode"] = args.agg_mode
    if args.schema:
        fe_cfg["schema"] = args.schema
    mode = fe_cfg.get("agg_mode", "causal")
    tag = dataset_tag(mode, fe_cfg.get("schema", "full"))

    src = Path("data/processed")
    train_df = pd.read_parquet(src / "train_raw.parquet")
    val_df = pd.read_parquet(src / "val_raw.parquet")
    test_df = pd.read_parquet(src / "test_raw.parquet")
    logger.info("Loaded raw splits: train {} val {} test {}", len(train_df), len(val_df), len(test_df))

    data, meta = run_feature_pipeline_v2(train_df, val_df, test_df, fe_cfg)
    del train_df, val_df, test_df
    import gc; gc.collect()

    out_dir = Path(cfg["paths"]["processed_v2"].format(mode=tag))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in list(data):                         # write and free one split at a time
        data.pop(name).to_parquet(out_dir / f"{name}.parquet", index=False)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
    logger.info("Saved v2 features ({} cols) to {}", len(meta["feature_cols"]), out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agg-mode", choices=["causal", "transductive"], default=None)
    p.add_argument("--schema", choices=["full", "raw"], default=None,
                   help="raw = only real-world gateway fields (no Vesta V/C/D2-15/id_01-11)")
    main(p.parse_args())

"""
Phase 9 (optional) — does a learned sequence score add anything on top of the v2 features?

  python scripts/run_sequence_experiment.py [--device gpu] [--epochs 3]

1. time-sliced OOF GRU score for every train row, GRU-on-all-train score for val/test
2. LightGBM on the time folds WITH vs WITHOUT `seq_gru_score`
3. keep rule (>= +0.005 mean CV AUC-PR, >= 2/3 folds) → appended to docs/ABLATIONS.md
The val holdout is not used for the decision. Expected gain at this scale: 0 to +0.01.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import average_precision_score

from src.experiments.sequence_embedding import build_sequences, time_sliced_oof, train_score
from src.feature_engineering.pipeline_v2 import dataset_tag
from src.labels.policy import LabelPolicy
from src.training.evaluation import make_folds
from src.utils.run_guard import finish
from src.training.gbdt import fit_model, predict


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    tcfg = cfg["training_v2"]
    tag = dataset_tag(args.agg_mode or cfg["fe_v2"]["agg_mode"], args.schema or cfg["fe_v2"].get("schema", "full"))
    fdir = Path(cfg["paths"]["processed_v2"].format(mode=tag))
    out = Path(cfg["paths"]["artifacts_v2"].format(mode=tag)); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((fdir / "meta.json").read_text())
    feats = meta["feature_cols"]
    parts = {s: pd.read_parquet(fdir / f"{s}.parquet") for s in ["train", "val", "test"]}
    allrows = pd.concat(parts.values(), ignore_index=True)
    X, M = build_sequences(allrows, k=args.k)                 # history crosses split boundaries (causal)
    n_tr, n_va = len(parts["train"]), len(parts["val"])
    tr = parts["train"]; y = tr["isFraud"].to_numpy().astype(int)

    logger.info("Time-sliced OOF GRU scores for train rows…")
    seq_tr = time_sliced_oof(X[:n_tr], M[:n_tr], y, tr["day"].to_numpy(), epochs=args.epochs, device=args.device)
    logger.info("GRU on all train → val/test scores…")
    seq_rest = train_score(X[:n_tr], M[:n_tr], y, X[n_tr:], M[n_tr:], epochs=args.epochs, device=args.device)
    pd.DataFrame({"TransactionID": allrows["TransactionID"], "seq_gru_score": np.r_[seq_tr, seq_rest]}).to_parquet(
        out / "seq_gru_score.parquet", index=False)

    tr = tr.assign(seq_gru_score=seq_tr)
    policy = LabelPolicy.from_cfg(meta.get("label_policy") or cfg.get("label_policy"))
    folds = make_folds(tr["TransactionDT"].to_numpy(), y, tr["TransactionID"].to_numpy(), tcfg["n_folds"],
                       tcfg["fold_val_days"], policy, es_frac=tcfg.get("es_frac", 0.5))
    params = dict(tcfg["lgbm"]); params["learning_rate"] = args.lr
    res = {"base": [], "with_seq": []}
    for fo in folds:
        for name, cols in [("base", feats), ("with_seq", feats + ["seq_gru_score"])]:
            m, _ = fit_model("lgbm", params, tr.iloc[fo.train][cols], fo.y_train, tr.iloc[fo.es][cols], y[fo.es], 100,
                             device=args.device)                       # ES rows are not scored
            res[name].append(average_precision_score(y[fo.score], predict("lgbm", m, tr.iloc[fo.score][cols])))
    gain = np.array(res["with_seq"]) - np.array(res["base"])
    keep = gain.mean() >= 0.005 and (gain > 0).sum() >= 2
    line = (f"\n## {dt.datetime.now():%Y-%m-%d %H:%M} — Phase 9 GRU sequence score ({tag})\n\n"
            f"base CV AP {np.mean(res['base']):.4f} | with GRU score {np.mean(res['with_seq']):.4f} | "
            f"gain {gain.mean():+.4f} (folds {np.round(gain, 4).tolist()}) → **{'KEEP' if keep else 'DROP'}**\n")
    with open("docs/ABLATIONS.md", "a", encoding="utf-8") as f:
        f.write(line)
    (out / "seq_experiment.json").write_text(json.dumps({
        "base_cv_ap": float(np.mean(res["base"])), "with_seq_cv_ap": float(np.mean(res["with_seq"])),
        "gain": float(gain.mean()), "fold_gains": gain.round(5).tolist(), "verdict": "KEEP" if keep else "DROP"}, indent=1))
    print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg-mode", choices=["causal", "transductive"], default=None)
    ap.add_argument("--schema", choices=["full", "raw"], default=None)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    main(ap.parse_args())
    finish()

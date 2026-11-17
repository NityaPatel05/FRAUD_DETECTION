"""
Phase 6 — held-out fraud pattern test (Monzo's method).

"Does the system catch fraud it was never trained on?" Remove every FRAUD row of one
pattern (default: each ProductCD in turn) from training, train LightGBM, and measure
how much of that pattern's fraud is still caught on the val holdout — compared with
a model that did see it. Run it before and after adding feature families to show
whether they improve generalisation to NEW fraud, which is what the VAE was for.

  python scripts/run_heldout_pattern_test.py [--column ProductCD] [--schema raw]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import average_precision_score, roc_curve

from src.feature_engineering.pipeline_v2 import dataset_tag
from src.labels.policy import LabelPolicy, training_target
from src.training.evaluation import make_folds
from src.utils.run_guard import finish
from src.training.gbdt import fit_model, predict


def recall_of_subset_at_fpr(y, p, subset, fpr=0.01):
    """Threshold set at `fpr` on ALL legit rows; recall measured on frauds in `subset`."""
    f, t, thr = roc_curve(y, p)
    th = thr[np.searchsorted(f, fpr, side="right") - 1]
    sel = subset & (y == 1)
    return float((p[sel] >= th).mean()) if sel.any() else float("nan")


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    tcfg = cfg["training_v2"]
    mode = args.agg_mode or cfg["fe_v2"]["agg_mode"]
    tag = dataset_tag(mode, args.schema or cfg["fe_v2"].get("schema", "full"))
    d = Path(cfg["paths"]["processed_v2"].format(mode=tag))
    meta = json.loads((d / "meta.json").read_text())
    feats = meta["feature_cols"]
    train = pd.read_parquet(d / "train.parquet"); val = pd.read_parquet(d / "val.parquet")
    labels = meta["label_maps"].get(args.column)
    params = dict(tcfg["lgbm"]); params["learning_rate"] = args.lr
    dev = args.device or tcfg.get("device", "cpu")

    # rounds: early-stop once on the last train fold (val never used for this)
    # one label policy: targets as known at the cutoff, early stopping on its own rows
    policy = LabelPolicy.from_cfg(meta.get("label_policy") or cfg.get("label_policy"))
    y_true = train["isFraud"].to_numpy().astype(int)
    folds = make_folds(train["TransactionDT"].to_numpy(), y_true, train["TransactionID"].to_numpy(),
                       tcfg["n_folds"], tcfg["fold_val_days"], policy, es_frac=tcfg.get("es_frac", 0.5))
    f = folds[-1]
    _, rounds = fit_model("lgbm", params, train.iloc[f.train][feats], f.y_train, train.iloc[f.es][feats], y_true[f.es],
                          100, device=dev)
    cutoff = int(val["TransactionDT"].min())
    tgt, usable = training_target(y_true, train["TransactionID"].to_numpy(), train["TransactionDT"].to_numpy(),
                                  cutoff, policy)
    train = train[usable].reset_index(drop=True)
    y_tr = tgt[usable].astype(int)
    y_val = val["isFraud"].to_numpy().astype(int)

    full_model, _ = fit_model("lgbm", params, train[feats], y_tr, n_rounds=rounds, device=dev)
    p_full = predict("lgbm", full_model, val[feats])

    rows = []
    for code in sorted(val[args.column].dropna().unique()):
        name = labels[int(code)] if labels else code
        held_val = (val[args.column] == code).to_numpy()
        if (y_val[held_val] == 1).sum() < 30:
            continue
        keep = ~((train[args.column] == code).to_numpy() & (y_tr == 1))     # drop that pattern's FRAUD only
        m, _ = fit_model("lgbm", params, train.loc[keep, feats], y_tr[keep], n_rounds=rounds, device=dev)
        p = predict("lgbm", m, val[feats])
        r_seen = recall_of_subset_at_fpr(y_val, p_full, held_val)
        r_unseen = recall_of_subset_at_fpr(y_val, p, held_val)
        rows.append({"held_out": f"{args.column}={name}", "val_frauds": int((y_val[held_val] == 1).sum()),
                     "recall@1%FPR_seen": r_seen, "recall@1%FPR_unseen": r_unseen,
                     "retained": r_unseen / r_seen if r_seen else float("nan"),
                     "AP_subset_seen": average_precision_score(y_val[held_val], p_full[held_val]),
                     "AP_subset_unseen": average_precision_score(y_val[held_val], p[held_val])})
        logger.info("{}: recall@1%FPR seen {:.3f} → unseen {:.3f}", rows[-1]["held_out"], r_seen, r_unseen)

    res = pd.DataFrame(rows)
    out = Path(cfg["paths"]["artifacts_v2"].format(mode=tag)); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"heldout_{args.column}.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agg-mode", choices=["causal", "transductive"], default=None)
    p.add_argument("--device", choices=["cpu", "gpu"], default=None)
    p.add_argument("--schema", choices=["full", "raw"], default=None)
    p.add_argument("--column", default="ProductCD")
    p.add_argument("--lr", type=float, default=0.05)
    main(p.parse_args())
    finish()

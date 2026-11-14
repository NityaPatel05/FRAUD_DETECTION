"""
Feature-family ablation on the walk-forward time folds (val holdout untouched).

  python scripts/run_ablation.py                       # leave-one-family-out vs full set
  python scripts/run_ablation.py --families behaviour_v3 network_v4
  python scripts/run_ablation.py --agg-mode causal --schema raw

For every family recorded in meta.json["feature_families"], LightGBM is trained on
the same folds with and without that family. KEEP RULE (docs/IMPROVEMENT_PLAN.md):
removing the family must LOWER mean CV AUC-PR by >= 0.005, in >= 2 of 3 folds, and
recall@1%FPR must not improve when it is removed. Results are appended to
docs/ABLATIONS.md and saved as CSV.
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

from src.training.evaluation import time_folds, recall_at_fpr
from src.training.gbdt import fit_model, predict

MIN_GAIN = 0.005


def cv_scores(train, feats, params, folds, es, device="cpu"):
    y = train["isFraud"].to_numpy().astype(int)
    aps, recs = [], []
    for tr, va in folds:
        m, _ = fit_model("lgbm", params, train.iloc[tr][feats], y[tr],
                         train.iloc[va][feats], y[va], early_stopping_rounds=es, device=device)
        p = predict("lgbm", m, train.iloc[va][feats])
        aps.append(average_precision_score(y[va], p))
        recs.append(recall_at_fpr(y[va], p, 0.01))
    return np.array(aps), np.array(recs)


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    tcfg = cfg["training_v2"]
    mode = args.agg_mode or cfg["fe_v2"]["agg_mode"]
    schema = args.schema or cfg["fe_v2"].get("schema", "full")
    from src.feature_engineering.pipeline_v2 import dataset_tag
    tag = dataset_tag(mode, schema)
    feat_dir = Path(cfg["paths"]["processed_v2"].format(mode=tag))
    meta = json.loads((feat_dir / "meta.json").read_text())
    train = pd.read_parquet(feat_dir / "train.parquet")

    dev = args.device or tcfg.get("device", "cpu")
    families = meta["feature_families"]
    todo = args.families or [f for f in families if f != "raw_columns"]
    feats_all = list(meta["feature_cols"])
    params = dict(tcfg["lgbm"]); params["learning_rate"] = args.lr; params["n_estimators"] = 4000
    folds = time_folds(train["day"].to_numpy(), tcfg["n_folds"], tcfg["fold_val_days"], tcfg["gap_days"])

    logger.info("Baseline: all {} features", len(feats_all))
    base_ap, base_rec = cv_scores(train, feats_all, params, folds, args.es, dev)
    rows = [{"family": "(all features)", "n_removed": 0, "ap_mean": base_ap.mean(), "ap_std": base_ap.std(),
             "delta_if_removed": 0.0, "folds_hurt_by_removal": "-", "recall1fpr_mean": base_rec.mean(),
             "verdict": "baseline"}]
    for fam in todo:
        cols = set(families.get(fam, []))
        if not cols:
            logger.warning("family {} empty/missing — skipped", fam); continue
        feats = [c for c in feats_all if c not in cols]
        ap, rec = cv_scores(train, feats, params, folds, args.es, dev)
        delta = base_ap - ap                       # >0 means the family helps
        hurt = int((delta > 0).sum())
        keep = delta.mean() >= MIN_GAIN and hurt >= 2 and rec.mean() <= base_rec.mean() + 1e-9
        rows.append({"family": fam, "n_removed": len(cols), "ap_mean": ap.mean(), "ap_std": ap.std(),
                     "delta_if_removed": float(delta.mean()), "folds_hurt_by_removal": f"{hurt}/{len(folds)}",
                     "recall1fpr_mean": rec.mean(), "verdict": "KEEP" if keep else "review/drop"})
        logger.info("{:<18} removed {:>4} cols | AP {:.4f} | gain from family {:+.4f} ({}/{}) → {}",
                    fam, len(cols), ap.mean(), delta.mean(), hurt, len(folds), rows[-1]["verdict"])

    res = pd.DataFrame(rows)
    out_dir = Path(cfg["paths"]["artifacts_v2"].format(mode=tag)); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    res.to_csv(out_dir / f"ablation_{dt.datetime.now():%Y%m%d_%H%M}.csv", index=False)
    md = Path("docs/ABLATIONS.md")
    header = "" if md.exists() else "# Ablation log\n\nKeep rule: family must add >= +0.005 mean CV AUC-PR, in >= 2/3 folds.\n"
    with open(md, "a", encoding="utf-8") as f:
        f.write(f"{header}\n## {stamp} — agg_mode={mode}, schema={schema}, lr={args.lr}\n\n")
        f.write(res.to_markdown(index=False, floatfmt=".4f") + "\n")
    print(res.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agg-mode", choices=["causal", "transductive"], default=None)
    p.add_argument("--device", choices=["cpu", "gpu"], default=None)
    p.add_argument("--schema", choices=["full", "raw"], default=None)
    p.add_argument("--families", nargs="+", default=None)
    p.add_argument("--lr", type=float, default=0.05, help="fast learning rate for ablations")
    p.add_argument("--es", type=int, default=100, help="early-stopping rounds")
    main(p.parse_args())

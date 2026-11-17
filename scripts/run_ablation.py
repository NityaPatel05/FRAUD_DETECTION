"""
Feature-family ablation on the rolling time folds (val holdout untouched).

  python scripts/run_ablation.py                       # leave-one-family-out + the overlapping-families group
  python scripts/run_ablation.py --families behaviour_v3 network_v4
  python scripts/run_ablation.py --groups behaviour,behaviour_v3,hmm_sequence,graph,network_v4
  python scripts/run_ablation.py --groups overlap      # preset name for the group above
  python scripts/run_ablation.py --seeds 5 --agg-mode causal --schema raw

For every family (or comma-separated GROUP of families removed together) in
meta.json["feature_families"], LightGBM is trained on the same folds with and without it.

Noise control: run-to-run noise of a single fit is about +-0.003 AUC-PR, the old keep
threshold was 0.005. So every arm is fitted with ``--seeds N`` seeds (default 3) on every
fold and the seeds are averaged; the comparison is PAIRED per fold (same folds, same
seeds): delta_f = AP_full,f - AP_without,f. We report mean delta +- standard error over folds
and the verdict is  KEEP  only if  mean delta >= 0.005  AND  mean delta > 2 x SE.

Families overlap (behaviour, behaviour_v3, hmm_sequence, graph and network_v4 all encode the
same client/entity history): removing one at a time understates each of them because the
others cover for it. The preset group removes them ALL, which tells you whether the whole
family of history features earns its place.

Early stopping uses the first half of each val window, scoring the second half (never the
same rows). Targets are the labels known at each fold's cutoff (src/labels/policy.py).
Results are appended to docs/ABLATIONS.md and saved as CSV.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import average_precision_score

from src.labels.policy import LabelPolicy
from src.training.evaluation import make_folds, recall_at_fpr, mean_se
from src.training.gbdt import fit_model, predict
from src.utils.run_guard import finish

MIN_GAIN = 0.005
SE_MULT = 2.0
PRESET_GROUPS = {"overlap": ["behaviour", "behaviour_v3", "hmm_sequence", "graph", "network_v4"]}


def ablation_verdict(delta_per_fold: Sequence[float], min_gain: float = MIN_GAIN,
                     se_mult: float = SE_MULT) -> Tuple[float, float, str]:
    """Paired per-fold deltas (full minus without, >0 = the removed columns help).

    -> (mean delta, standard error, verdict). KEEP needs mean >= min_gain AND mean > se_mult * SE.
    """
    mean, se = mean_se(list(delta_per_fold))
    if not np.isfinite(mean):
        return mean, se, "n/a"
    if mean >= min_gain and mean > se_mult * se:
        return mean, se, "KEEP"
    if mean < -se_mult * se and mean < 0:
        return mean, se, "HURTS (removing it helps)"
    if mean >= min_gain:
        return mean, se, "inconclusive (gain within noise)"
    return mean, se, "drop (gain below threshold)"


def parse_groups(items: Sequence[str]) -> Dict[str, List[str]]:
    """'a,b,c' -> {'a+b+c': [a,b,c]};  a preset name ('overlap') expands to its families."""
    out = {}
    for it in items:
        fams = PRESET_GROUPS.get(it) or [x for x in it.split(",") if x]
        out["+".join(fams)] = list(fams)
    return out


def cv_scores(train, feats, params, folds, es, device="cpu", seeds=1):
    """Per-fold AP / recall@1%FPR averaged over seeds, on the SCORING part of each fold."""
    y = train["isFraud"].to_numpy().astype(int)
    X = train[feats]
    aps = np.zeros((seeds, len(folds)))
    recs = np.zeros((seeds, len(folds)))
    for s in range(seeds):
        for i, f in enumerate(folds):
            m, _ = fit_model("lgbm", params, X.iloc[f.train], f.y_train, X.iloc[f.es], y[f.es],
                             early_stopping_rounds=es, seed=42 + s, device=device)
            p = predict("lgbm", m, X.iloc[f.score])
            aps[s, i] = average_precision_score(y[f.score], p)
            recs[s, i] = recall_at_fpr(y[f.score], p, 0.01)
    return aps.mean(axis=0), recs.mean(axis=0)


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
    feats_all = list(meta["feature_cols"])
    params = dict(tcfg["lgbm"]); params["learning_rate"] = args.lr; params["n_estimators"] = 4000
    policy = LabelPolicy.from_cfg(meta.get("label_policy") or cfg.get("label_policy"))
    y = train["isFraud"].to_numpy().astype(int)
    folds = make_folds(train["TransactionDT"].to_numpy(), y, train["TransactionID"].to_numpy(), tcfg["n_folds"],
                       tcfg["fold_val_days"], policy, es_frac=tcfg.get("es_frac", 0.5))

    # arms: name -> families removed together
    arms: Dict[str, List[str]] = {}
    if args.groups:
        arms.update(parse_groups(args.groups))
    if args.families or not args.groups:
        for f in (args.families or [f for f in families if f != "raw_columns"]):
            arms[f] = [f]
        if not args.families:                         # default run also tests the overlapping history families
            arms.update(parse_groups(["overlap"]))

    logger.info("Baseline: all {} features | {} folds x {} seeds per arm", len(feats_all), len(folds), args.seeds)
    base_ap, base_rec = cv_scores(train, feats_all, params, folds, args.es, dev, args.seeds)
    rows = [{"family": "(all features)", "n_removed": 0, "ap_mean": base_ap.mean(), "ap_std": base_ap.std(),
             "delta_if_removed": 0.0, "delta_se": 0.0, "folds_hurt_by_removal": "-",
             "recall1fpr_mean": base_rec.mean(), "verdict": "baseline"}]
    for name, fams in arms.items():
        cols = set(c for fam in fams for c in families.get(fam, []))
        missing = [fam for fam in fams if not families.get(fam)]
        if not cols:
            logger.warning("{} empty/missing - skipped", name); continue
        if missing:
            logger.warning("{}: families {} empty/missing (ignored)", name, missing)
        feats = [c for c in feats_all if c not in cols]
        ap, rec = cv_scores(train, feats, params, folds, args.es, dev, args.seeds)
        delta = base_ap - ap                       # paired per fold; >0 means the removed columns help
        mean, se, verdict = ablation_verdict(delta)
        hurt = int((delta > 0).sum())
        rows.append({"family": name, "n_removed": len(cols), "ap_mean": ap.mean(), "ap_std": ap.std(),
                     "delta_if_removed": mean, "delta_se": se, "folds_hurt_by_removal": f"{hurt}/{len(folds)}",
                     "recall1fpr_mean": rec.mean(), "verdict": verdict})
        logger.info("{:<40} removed {:>4} cols | AP {:.4f} | gain from it {:+.4f} +- {:.4f} ({}/{} folds) -> {}",
                    name, len(cols), ap.mean(), mean, se, hurt, len(folds), verdict)

    res = pd.DataFrame(rows)
    out_dir = Path(cfg["paths"]["artifacts_v2"].format(mode=tag)); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    res.to_csv(out_dir / f"ablation_{dt.datetime.now():%Y%m%d_%H%M}.csv", index=False)
    md = Path("docs/ABLATIONS.md")
    header = "" if md.exists() else (f"# Ablation log\n\nKeep rule: mean paired per-fold gain >= {MIN_GAIN} AND > {SE_MULT:g} x "
                                     "standard error (seeds averaged).\n")
    with open(md, "a", encoding="utf-8") as f:
        f.write(f"{header}\n## {stamp} - agg_mode={mode}, schema={schema}, lr={args.lr}, seeds={args.seeds}, "
                f"folds={len(folds)}\n\n")
        f.write(md_table(res) + "\n")
    print(res.to_string(index=False))


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without pandas' optional `tabulate` dependency."""
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agg-mode", choices=["causal", "transductive"], default=None)
    p.add_argument("--device", choices=["cpu", "gpu"], default=None)
    p.add_argument("--schema", choices=["full", "raw"], default=None)
    p.add_argument("--families", nargs="+", default=None, help="remove each of these families on its own")
    p.add_argument("--groups", nargs="+", default=None,
                   help="remove groups of families together: 'a,b,c' (comma list) or preset 'overlap' "
                        "= behaviour,behaviour_v3,hmm_sequence,graph,network_v4")
    p.add_argument("--seeds", type=int, default=3, help="seeds per arm, averaged (paired across arms)")
    p.add_argument("--lr", type=float, default=0.05, help="fast learning rate for ablations")
    p.add_argument("--es", type=int, default=100, help="early-stopping rounds")
    main(p.parse_args())
    finish()

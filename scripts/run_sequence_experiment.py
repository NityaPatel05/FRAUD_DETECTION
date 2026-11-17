"""
Phase D - do self-supervised sequence embeddings + "surprise" add anything to the v2 GBDT?

  python scripts/run_sequence_experiment.py --device cpu                  # dev (use a --sample-frac 0.3 dataset)
  python scripts/run_sequence_experiment.py --device cuda --seeds 3       # GPU machine, full data

1. pretrain a GRU next-transaction model on LABEL-FREE train rows (default: rows before the first CV cutoff,
   so every CV validation window is out-of-sample for the encoder too)
2. causal embedding + surprise features for train/val/test (history crosses split boundaries)
3. LightGBM on the project's rolling folds: base vs +seq (emb+surprise) vs +surprise-only vs +emb-only,
   >=3 seeds, paired per-fold deltas (seed-averaged), keep rule from scripts/run_ablation.py
4. slices: all / cold_start (client never seen in the fold's train) / copied_removed; VDR@1%FPR
Outputs: artifacts/v2/<tag>/seq_features.parquet, seq_encoder.pt, seq_experiment.json ; appends docs/ABLATIONS.md
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import average_precision_score

from src.feature_engineering.pipeline_v2 import dataset_tag
from src.labels.policy import LabelPolicy
from src.sequence import FEATURE_PREFIX, SURPRISE_COLS, SeqConfig, SeqSpec, SequenceEncoder, client_order
from src.sequence.data import gaps
from src.training.evaluation import (copied_positive_flags, make_folds, mean_se, recall_at_fpr, slice_masks,
                                     value_detection_rate)
from src.training.gbdt import fit_model, predict
from src.utils.run_guard import finish

DAY = 86400


def keep_verdict(deltas):
    from run_ablation import ablation_verdict
    return ablation_verdict(deltas)


def ap(y, p, m):
    m = np.asarray(m, dtype=bool)
    return float(average_precision_score(y[m], p[m])) if y[m].sum() >= 5 and y[m].sum() < m.sum() else np.nan


def main(a):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    tcfg, scfg = cfg["training_v2"], cfg.get("sequence", {})
    tag = dataset_tag(a.agg_mode or cfg["fe_v2"]["agg_mode"], a.schema or cfg["fe_v2"].get("schema", "full"))
    fdir = Path(cfg["paths"]["processed_v2"].format(mode=tag))
    out = Path(cfg["paths"]["artifacts_v2"].format(mode=tag)); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((fdir / "meta.json").read_text())
    feats = meta["feature_cols"]
    torch_dev = "cuda" if a.device in ("cuda", "gpu") else "cpu"
    gbdt_dev = "gpu" if torch_dev == "cuda" else "cpu"
    t_all = time.time()

    parts = [pd.read_parquet(fdir / f"{s}.parquet") for s in ["train", "val", "test"]]
    n_tr = len(parts[0])
    df = pd.concat(parts, ignore_index=True)
    train = df.iloc[:n_tr].reset_index(drop=True)
    y = train["isFraud"].to_numpy().astype(int)
    policy = LabelPolicy.from_cfg(meta.get("label_policy") or cfg.get("label_policy"))
    folds = make_folds(train["TransactionDT"].to_numpy(), y, train["TransactionID"].to_numpy(), tcfg["n_folds"],
                       tcfg["fold_val_days"], policy, es_frac=tcfg.get("es_frac", 0.5))

    # ---- 1. label-free pretraining window
    client = df["uid1_code"].to_numpy()
    tt, tid = df["TransactionDT"].to_numpy(), df["TransactionID"].to_numpy()
    pre = np.zeros(len(df), dtype=bool)
    pre[:n_tr] = True if a.pretrain == "train" else (train["day"].to_numpy() < folds[0].val_start_day)
    pdf = df[pre].reset_index(drop=True)
    o, _, st = client_order(client[pre], tt[pre], tid[pre])
    g = np.empty(len(pdf)); g[o] = gaps(tt[pre][o], st)
    spec = SeqSpec.fit(pdf, g, cat_cols=scfg.get("cat_cols", ["ProductCD", "card4", "card6", "P_emaildomain",
                                                            "DeviceType", "addr1", "addr2", "DeviceInfo"]))
    cfgm = SeqConfig(hidden=scfg.get("hidden", 32), emb_dim=scfg.get("emb_dim", 12), epochs=a.epochs or scfg.get("epochs", 4),
                     max_len=scfg.get("max_len", 96), lr=scfg.get("lr", 3e-3), seed=a.seed)
    enc = SequenceEncoder(spec, cfgm, torch_dev)
    logger.info("pretraining on {} label-free rows ({} cats: {})", len(pdf), len(spec.cat_cols), spec.cat_cols)
    t0 = time.time()
    enc.fit(spec.encode(pdf), client[pre], tid[pre], log=logger.info)
    t_pre = time.time() - t0
    enc.save(str(out / "seq_encoder.pt"))

    # ---- 2. causal features for all rows
    t0 = time.time()
    sf = enc.transform(spec.encode(df), client, tid)
    t_feat = time.time() - t0
    seq_cols = list(sf.columns)
    emb_cols = [c for c in seq_cols if c.startswith(FEATURE_PREFIX)]
    pd.concat([df[["TransactionID"]], sf], axis=1).to_parquet(out / "seq_features.parquet", index=False)
    trf = pd.concat([train, sf.iloc[:n_tr].reset_index(drop=True)], axis=1)

    # ---- 3. CV
    arms = {"base": feats, "seq_all": feats + seq_cols, "surprise_only": feats + SURPRISE_COLS,
            "emb_only": feats + emb_cols}
    params = dict(tcfg["lgbm"]); params["learning_rate"] = a.lr; params["n_estimators"] = a.n_estimators
    cp = copied_positive_flags(trf["uid1_code"].to_numpy(), trf["TransactionDT"].to_numpy(),
                               trf["TransactionID"].to_numpy(), y)
    amt = trf["TransactionAmt"].to_numpy()
    cl = trf["uid1_code"].to_numpy()
    metrics = ["ap_all", "ap_cold_start", "ap_copied_removed", "vdr_1pct", "recall_1pct"]
    R = {arm: np.full((a.seeds, len(folds), len(metrics)), np.nan) for arm in arms}
    t0 = time.time()
    for s in range(a.seeds):
        for i, f in enumerate(folds):
            masks = slice_masks(cl[f.score], cl[f.train], cp[f.score])
            for arm, cols in arms.items():
                X = trf[cols]
                m, _ = fit_model("lgbm", params, X.iloc[f.train], f.y_train, X.iloc[f.es], y[f.es],
                                 early_stopping_rounds=a.es, seed=42 + s, device=gbdt_dev)
                p = predict("lgbm", m, X.iloc[f.score])
                ys = y[f.score]
                R[arm][s, i] = [ap(ys, p, masks["all"]), ap(ys, p, masks["cold_start"]),
                                ap(ys, p, masks["copied_removed"]),
                                value_detection_rate(ys, p, amt[f.score], 0.01), recall_at_fpr(ys, p, 0.01)]
            logger.info("seed {} fold {} done ({:.0f}s)", s, f.k, time.time() - t0)
    t_cv = time.time() - t0

    res = {"tag": tag, "pretrain": a.pretrain, "n_pretrain_rows": int(len(pdf)), "seeds": a.seeds,
           "n_folds": len(folds), "arms": {}, "seconds": {"pretrain": t_pre, "features": t_feat, "cv": t_cv}}
    base = R["base"].mean(0)                                  # [fold, metric], seed-averaged
    lines = []
    for arm in arms:
        cur = R[arm].mean(0)
        d = {}
        for k, mn in enumerate(metrics):
            delta = cur[:, k] - base[:, k]
            ok = np.isfinite(delta)
            mu, se = mean_se(delta[ok]) if ok.any() else (np.nan, np.nan)
            d[mn] = {"mean": float(np.nanmean(cur[:, k])), "base": float(np.nanmean(base[:, k])),
                     "delta": mu, "se": se, "fold_deltas": np.round(delta, 5).tolist()}
        if arm != "base":
            mu, se, verdict = keep_verdict(cur[:, 0] - base[:, 0])
            d["verdict"] = verdict
            lines.append(f"{arm}: AP {d['ap_all']['mean']:.4f} vs base {d['ap_all']['base']:.4f} "
                         f"delta {mu:+.4f} +/- {se:.4f} (SE, {len(folds)} folds x {a.seeds} seeds) -> {verdict}")
        res["arms"][arm] = d

    # ---- 4. serving numbers
    sample = spec.encode(df.iloc[:2000])
    stt = enc.init_state()
    t0 = time.time()
    for i in range(500):
        _, _, stt = enc.step(stt, sample.row(i))
    res["serving"] = {"step_ms_cpu_1txn": (time.time() - t0) / 500 * 1000, "state_bytes_per_client": int(stt.nbytes),
                      "n_clients_in_data": int(len(np.unique(client[client >= 0])))}
    res["seconds"]["total"] = time.time() - t_all
    (out / "seq_experiment.json").write_text(json.dumps(res, indent=1))
    summ = (f"\n## {dt.datetime.now():%Y-%m-%d %H:%M} - Phase D sequence embeddings ({tag}, pretrain={a.pretrain}, "
            f"{a.seeds} seeds)\n\n" + "\n".join(f"- {l}" for l in lines) + "\n")
    with open("docs/ABLATIONS.md", "a", encoding="utf-8") as fh:
        fh.write(summ)
    print(summ)
    print(json.dumps({k: {m: (round(v[m]["delta"], 4), round(v[m]["se"], 4)) for m in metrics}
                      for k, v in res["arms"].items() if k != "base"}, indent=1))
    print(res["serving"], res["seconds"])


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--agg-mode", choices=["causal", "transductive"], default=None)
    ap_.add_argument("--schema", choices=["full", "raw"], default=None)
    ap_.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    ap_.add_argument("--pretrain", choices=["first_cutoff", "train"], default="first_cutoff")
    ap_.add_argument("--epochs", type=int, default=None)
    ap_.add_argument("--seeds", type=int, default=3)
    ap_.add_argument("--seed", type=int, default=0, help="encoder init seed")
    ap_.add_argument("--lr", type=float, default=0.05, help="LightGBM learning rate")
    ap_.add_argument("--n-estimators", type=int, default=2000)
    ap_.add_argument("--es", type=int, default=100)
    main(ap_.parse_args())
    finish()

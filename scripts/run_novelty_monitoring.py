"""
Phase 6 — novelty monitoring over the validation period, simulated as if live.

  python scripts/run_novelty_monitoring.py [--agg-mode causal] [--schema full]

1. Attack detection on hourly segment series (train+val history, baseline = past only)
2. For the strongest alerts: mine contrast rules (alert window vs previous 7 days)
3. Backtest each candidate rule on labels that were MATURE at alert time, and
   (offline only) on the alert window's eventual labels
4. Save candidates as SHADOW rules with an expiry → artifacts/v2/novelty/candidate_rules.yaml
5. Drift report (PSI + domain classifier) train vs val on the v2 features
6. Unusual-cluster discovery for analysts (Isolation Forest + KMeans) on val

Outputs: artifacts/v2/novelty/{alerts.csv, candidate_rules.yaml, rule_backtests.csv,
psi.csv, drift.json, discovery.csv}
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

from src.feature_engineering.derived_signals import email_class
from src.novelty.attack_detection import add_amount_band, detect_attacks
from src.novelty.rule_mining import mine_contrast_rules, rule_to_conditions
from src.novelty.rules_engine import backtest, save_rules
from src.novelty.drift import psi_report, domain_classifier
from src.novelty.discovery import discover_unusual_clusters


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    ncfg = cfg["novelty"]
    out = Path(ncfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    from src.labels.policy import LabelPolicy, known_label
    policy = LabelPolicy.from_cfg(cfg.get("label_policy"))          # the one label policy

    # ── raw, human-readable columns for segments and rules ───────────────────
    cols = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"] + ncfg["segment_cols"] + ncfg["rule_cols"]
    cols = list(dict.fromkeys(c for c in cols if c not in ("amount_band", "P_email_class")))
    tr = pd.read_parquet("data/processed/train_raw.parquet", columns=[c for c in cols])
    va = pd.read_parquet("data/processed/val_raw.parquet", columns=[c for c in cols])
    df = pd.concat([tr, va], ignore_index=True).sort_values("TransactionDT").reset_index(drop=True)
    df["amount_band"] = add_amount_band(df)
    df["P_email_class"] = email_class(df["P_emaildomain"]).astype(str)
    val_start = va["TransactionDT"].min()

    # ── 1. attack detection (keep alerts that START in the val period) ──────
    alerts = detect_attacks(df, ncfg["segment_cols"], ncfg["period_sec"], ncfg["z_threshold"],
                            ncfg["min_periods"], ncfg["min_volume"], ncfg["baseline_periods"])
    alerts = alerts[alerts["start_time"] >= val_start].reset_index(drop=True)
    alerts.to_csv(out / "alerts.csv", index=False)
    logger.info("{} segment alerts in the validation period", len(alerts))

    # ── 2-4. mine → backtest → shadow rules ─────────────────────────────────
    rules, bts = [], []
    for k, a in alerts.head(ncfg["max_alerts_to_mine"]).iterrows():
        seg_rows = df[(df["TransactionDT"] >= a.start_time) & (df["TransactionDT"] < a.end_time)
                      & (df[a.segment].astype(str) == a.value)]
        base = df[(df["TransactionDT"] >= a.start_time - 7 * 86400) & (df["TransactionDT"] < a.start_time)]
        cand = mine_contrast_rules(seg_rows, base, ncfg["rule_cols"], ncfg["min_support"],
                                   ncfg["min_lift"], ncfg["max_rules_per_alert"], y_col="isFraud")
        kl = known_label(df["isFraud"].to_numpy(), df["TransactionID"].to_numpy(),
                         df["TransactionDT"].to_numpy(), a.start_time, policy)     # labels known at alert time
        matured = df.assign(isFraud=kl).dropna(subset=["isFraud"])
        matured = matured[matured["TransactionDT"] < a.start_time]
        after = df[(df["TransactionDT"] >= a.start_time) & (df["TransactionDT"] < a.start_time + ncfg["rule_ttl_days"] * 86400)]
        for j, c in cand.iterrows():
            rule = {"id": f"alert{k}_rule{j}", "conditions": rule_to_conditions(c.rule),
                    "action": "review", "mode": "shadow",
                    "expires_at": int(a.end_time + ncfg["rule_ttl_days"] * 86400),
                    "created_from": f"{a.segment}={a.value} z={a.peak_z:.1f} t={a.start_time}",
                    "lift": round(float(c.lift), 2)}
            hist = backtest(matured, rule)                    # what an analyst could see at alert time
            fut = backtest(after, rule)                       # offline evaluation only
            bts.append({**{f"hist_{k2}": v for k2, v in hist.items()},
                        **{f"next_{k2}": v for k2, v in fut.items() if k2 != "rule_id"},
                        "rule": c.rule, "lift": c.lift})
            if hist["hits"] == 0 or (hist["precision"] >= ncfg["min_backtest_precision"]):
                rules.append(rule)
    save_rules(rules, out / "candidate_rules.yaml")
    pd.DataFrame(bts).to_csv(out / "rule_backtests.csv", index=False)
    logger.info("{} candidate shadow rules saved (of {} mined)", len(rules), len(bts))

    # ── 5. drift on v2 features (train vs val) ──────────────────────────────
    from src.feature_engineering.pipeline_v2 import dataset_tag
    mode = args.agg_mode or cfg["fe_v2"]["agg_mode"]
    tag = dataset_tag(mode, args.schema or cfg["fe_v2"].get("schema", "full"))
    fdir = Path(cfg["paths"]["processed_v2"].format(mode=tag))
    if (fdir / "meta.json").exists():
        meta = json.loads((fdir / "meta.json").read_text())
        feats = meta["feature_cols"]
        ftr = pd.read_parquet(fdir / "train.parquet", columns=feats)
        fva = pd.read_parquet(fdir / "val.parquet", columns=feats)
        rep = psi_report(ftr, fva, feats); rep.to_csv(out / "psi.csv", index=False)
        top = rep.head(40)["feature"].tolist()
        dc = domain_classifier(ftr.sample(min(len(ftr), 100_000), random_state=0),
                               fva.sample(min(len(fva), 100_000), random_state=0), feats)
        (out / "drift.json").write_text(json.dumps({"domain_auc": dc["auc"], "top_drivers": dc["top_drivers"],
                                                    "psi_investigate": rep[rep.level == "investigate"]["feature"].tolist(),
                                                    "psi_top40": top}, indent=1))
        logger.info("Drift: domain-classifier AUC {:.3f}; {} features PSI>0.2", dc["auc"],
                    int((rep.level == "investigate").sum()))

        # ── 6. discovery for analysts (val period) ──────────────────────────
        num = [c for c in ncfg["discovery_num_cols"] if c in fva.columns]
        disc_df = pd.read_parquet(fdir / "val.parquet", columns=list(dict.fromkeys(num + ["isFraud"])))
        disc = discover_unusual_clusters(disc_df, num, cat_cols=[])
        disc.to_csv(out / "discovery.csv", index=False)
        logger.info("Discovery: {} unusual clusters written", len(disc))
    else:
        logger.warning("No v2 features at {} — run run_feature_engineering.py for drift/discovery", fdir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agg-mode", choices=["causal", "transductive"], default=None)
    p.add_argument("--schema", choices=["full", "raw"], default=None)
    main(p.parse_args())

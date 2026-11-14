"""
Phase 8 — daily monitoring report + alerts from the decision log.

  python scripts/run_monitoring_report.py [--labels data/processed/val_raw.parquet]

Reads decision_v2.decision_log; optional labels (any parquet with TransactionID, isFraud)
enable false-positive-rate / recall tracking. Writes artifacts/v2/monitoring/.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.feedback.monitoring import alerts, daily_report, load_decision_log


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    fcfg = cfg["feedback_v2"]
    log = load_decision_log(cfg["decision_v2"]["decision_log"])
    labels = None
    if args.labels:
        lab = pd.read_parquet(args.labels, columns=["TransactionID", "isFraud"])
        labels = lab.set_index("TransactionID")["isFraud"]
    rep = daily_report(log, labels, fcfg["psi_reference_days"])
    al = alerts(rep, fcfg["alert_bands"])
    out = Path(fcfg["monitoring_dir"]); out.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out / "daily_report.csv", index=False); al.to_csv(out / "alerts.csv", index=False)
    print(rep.tail(10).to_string(index=False))
    print(f"\n{len(al)} alerts" + ("" if al.empty else ":\n" + al.to_string(index=False)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None)
    main(ap.parse_args())

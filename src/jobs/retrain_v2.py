"""
Phase 8 — scheduled retrain for the v2 pipeline + a minimal model registry.

  python -m src.jobs.retrain_v2 [--device gpu] [--dry-run]

Steps (each is the same script a human would run, so the job adds no hidden logic):
  1. run_preprocessing → run_feature_engineering → run_feature_selection
  2. run_training (evaluation on the val holdout) → metrics_val.json
  3. register the new bundle as CHALLENGER (never directly as champion)
The challenger then runs in shadow in the decision service; scripts/run_promotion_gate.py
decides promotion on matured labels. Retraining on immature labels is prevented by
using only labels older than feedback_v2.maturity_days (see src/feedback/labels.py)
when production labels are wired in.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml
from loguru import logger


class ModelRegistry:
    """registry.json: {"champion": {...}, "challenger": {...}, "history": [...]}"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"champion": None, "challenger": None, "history": []}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, default=str))

    def register_challenger(self, art_dir: str, metrics: Optional[Dict] = None) -> None:
        self.data["challenger"] = {"art_dir": art_dir, "registered_at": dt.datetime.now().isoformat(),
                                   "val_metrics": metrics}
        self.data["history"].append({"event": "challenger_registered", **self.data["challenger"]})
        self._save()

    def promote_challenger(self, evidence: Optional[Dict] = None) -> None:
        if not self.data.get("challenger"):
            raise ValueError("no challenger to promote")
        self.data["history"].append({"event": "promoted", "previous_champion": self.data.get("champion"),
                                     "at": dt.datetime.now().isoformat(), "evidence": evidence})
        self.data["champion"], self.data["challenger"] = self.data["challenger"], None
        self._save()


def run(cmd, dry):
    logger.info("$ {}", " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    py = sys.executable
    dev = ["--device", args.device] if args.device else []
    run([py, "scripts/run_preprocessing.py"], args.dry_run)
    run([py, "scripts/run_feature_engineering.py"], args.dry_run)
    run([py, "scripts/run_feature_selection.py"], args.dry_run)
    run([py, "scripts/run_training.py", *dev], args.dry_run)
    from src.feature_engineering.pipeline_v2 import dataset_tag
    tag = dataset_tag(cfg["fe_v2"]["agg_mode"], cfg["fe_v2"].get("schema", "full"))
    art = cfg["paths"]["artifacts_v2"].format(mode=tag)
    metrics_file = Path(art) / "metrics_val.json"
    metrics = json.loads(metrics_file.read_text()).get("blend_val") if metrics_file.exists() else None
    if not args.dry_run:
        ModelRegistry(cfg["feedback_v2"]["registry"]).register_challenger(art, metrics)
    logger.info("Registered challenger from {} — run it in shadow, then scripts/run_promotion_gate.py", art)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "gpu"], default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the steps without running them")
    main(ap.parse_args())

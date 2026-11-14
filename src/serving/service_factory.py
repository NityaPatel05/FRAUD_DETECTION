"""Build a ready DecisionService from config + trained artifacts (used by the API and benchmark)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from loguru import logger

from src.feature_engineering.pipeline_v2 import dataset_tag
from src.novelty.rules_engine import load_rules
from src.serving.decision import CostPolicy, DecisionLog, DecisionService, ModelBundle
from src.serving.online_features import OnlineFeatureEngine


def build_service(config_path: str = "configs/config.yaml", schema: Optional[str] = None,
                  warmup: bool = True, challenger_dir: Optional[str] = None) -> DecisionService:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    dcfg = dict(cfg["decision_v2"])
    tag = dataset_tag(cfg["fe_v2"]["agg_mode"], schema or cfg["fe_v2"].get("schema", "full"))
    feat_dir = Path(cfg["paths"]["processed_v2"].format(mode=tag))
    art_dir = Path(cfg["paths"]["artifacts_v2"].format(mode=tag))
    meta = json.loads((feat_dir / "meta.json").read_text())

    engine = OnlineFeatureEngine(meta)
    snap = Path(dcfg["state_snapshot"])
    if snap.exists():
        engine.load_state(snap)
        logger.info("Engine state loaded from {}", snap)
    elif warmup:
        logger.info("Warming engine state from history (train + val raw, labels with delay)…")
        hist = pd.concat([pd.read_parquet("data/processed/train_raw.parquet"),
                          pd.read_parquet("data/processed/val_raw.parquet")], ignore_index=True)
        engine.replay(hist, labels=True, collect=False)
        snap.parent.mkdir(parents=True, exist_ok=True)
        engine.save_state(snap)

    overrides = art_dir / "policy_overrides.json"
    if overrides.exists():
        dcfg.update(json.loads(overrides.read_text()))
    reg_path = Path(cfg.get("feedback_v2", {}).get("registry", ""))
    reg = json.loads(reg_path.read_text()) if reg_path.is_file() else {}
    champ_dir = (reg.get("champion") or {}).get("art_dir") or art_dir        # registry wins if present
    bundle = ModelBundle(champ_dir, dcfg["bundle_kind"])
    challenger_dir = challenger_dir or (reg.get("challenger") or {}).get("art_dir")
    challenger = ModelBundle(challenger_dir, "refit") if challenger_dir and challenger_dir != champ_dir else None
    rules = [r for r in load_rules(dcfg["rules_file"]) if r.get("mode") in ("shadow", "enforce")]
    lists_path = Path(dcfg["lists_file"])
    lists = yaml.safe_load(lists_path.read_text(encoding="utf-8")) if lists_path.exists() else None
    return DecisionService(engine, bundle, CostPolicy(dcfg), rules, lists,
                           DecisionLog(dcfg["decision_log"]), challenger,
                           use_client_smoothing=bundle.spec.get("use_client_smoothing", False))

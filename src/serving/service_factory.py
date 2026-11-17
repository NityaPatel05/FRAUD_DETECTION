"""Build a ready DecisionService from config + trained artifacts (used by the API and benchmark).

Served bundle: the RAW-schema one (decision_v2.serving_schema, default "raw") -- the only schema whose
inputs a real payment gateway sends. A full-schema bundle needs Vesta-only fields (V*, C*, D2-15,
id_01-11); it is refused for live serving unless ``allow_full_schema=True`` (replay benchmarks).
"""
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
from src.serving.online_features import OnlineFeatureEngine, SnapshotMismatchError


def build_service(config_path: str = "configs/config.yaml", schema: Optional[str] = None,
                  warmup: bool = True, challenger_dir: Optional[str] = None,
                  allow_full_schema: bool = False) -> DecisionService:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    dcfg = dict(cfg["decision_v2"])
    schema = schema or dcfg.get("serving_schema", "raw")
    if schema not in ("raw", "full"):
        raise ValueError(f"schema must be raw|full, got {schema}")
    if schema == "full" and not allow_full_schema:
        raise ValueError(
            "schema=full needs Vesta-only fields (V*, C*, D2-D15, id_01-id_11) that a real gateway does not "
            "send. Serve the raw-schema bundle (run_all with fe_v2.schema=raw, decision_v2.serving_schema: raw), "
            "or pass allow_full_schema=True for replay/benchmarks on the full IEEE columns.")
    tag = dataset_tag(cfg["fe_v2"]["agg_mode"], schema)
    feat_dir = Path(cfg["paths"]["processed_v2"].format(mode=tag))
    art_dir = Path(cfg["paths"]["artifacts_v2"].format(mode=tag))
    if not (feat_dir / "meta.json").exists():
        raise FileNotFoundError(f"{feat_dir / 'meta.json'} missing: build the {schema}-schema features and train first")
    meta = json.loads((feat_dir / "meta.json").read_text())

    overrides = art_dir / "policy_overrides.json"
    if overrides.exists():
        dcfg.update(json.loads(overrides.read_text()))
    reg_path = Path(cfg.get("feedback_v2", {}).get("registry", ""))
    reg = json.loads(reg_path.read_text()) if reg_path.is_file() else {}
    champ_dir = (reg.get("champion") or {}).get("art_dir") or art_dir        # registry wins if present
    bundle = ModelBundle(champ_dir, dcfg["bundle_kind"])
    if bundle.spec.get("tag") not in (None, tag):
        raise ValueError(f"champion bundle {champ_dir} was trained on feature set {bundle.spec['tag']!r} "
                         f"but the engine serves {tag!r}")
    challenger_dir = challenger_dir or (reg.get("challenger") or {}).get("art_dir")
    challenger = ModelBundle(challenger_dir, "refit") if challenger_dir and str(challenger_dir) != str(champ_dir) else None

    engine = OnlineFeatureEngine(meta, dcfg.get("state"))
    snap = Path(dcfg["state_snapshot"])
    expect = {"bundle": bundle.fingerprint()}
    loaded = False
    if snap.exists():
        try:
            engine.load_state(snap, expect_extra=expect)
            loaded = True
            logger.info("Engine state loaded from {} ({} txns indexed)", snap, len(engine.txn_keys))
        except SnapshotMismatchError as e:
            if not warmup:
                raise
            logger.warning("{} -- rebuilding state from history", e)
            engine = OnlineFeatureEngine(meta, dcfg.get("state"))
    if not loaded and warmup:
        logger.info("Warming engine state from history (train + val raw; fraud labels via the label policy)...")
        hist = pd.concat([pd.read_parquet("data/processed/train_raw.parquet"),
                          pd.read_parquet("data/processed/val_raw.parquet")], ignore_index=True)
        engine.replay(hist, labels=True, collect=False)
        snap.parent.mkdir(parents=True, exist_ok=True)
        engine.save_state(snap, extra=expect)

    rules = [r for r in load_rules(dcfg["rules_file"]) if r.get("mode") in ("shadow", "enforce")]
    lists_path = Path(dcfg["lists_file"])
    lists = yaml.safe_load(lists_path.read_text(encoding="utf-8")) if lists_path.exists() else None
    svc = DecisionService(engine, bundle, CostPolicy(dcfg), rules, lists,
                          DecisionLog(dcfg["decision_log"]), challenger,
                          use_client_smoothing=bundle.spec.get("use_client_smoothing", False))
    svc.snapshot_extra = expect
    return svc

"""
Phase 7 — decision service: features → model → rules → cost-based policy → action.

Layered decisioning as in Stripe Radar / Adyen / Sift / Nubank:
  1. hard lists      : allow-list → approve, block-list → decline (short-circuit)
  2. model           : calibrated fraud probability p (CalibratedRankBlender)
  3. rules           : ENFORCED rules can force review/step_up/block/allow ("allow" wins);
                       SHADOW rules are only logged (Phase 6 candidates)
  4. policy          : the action with the lowest EXPECTED COST given p and the amount
                       (Bahnsen-style cost-sensitive decision), review only above the
                       capacity-derived threshold
  5. reason codes    : top-3 TreeSHAP contributions → readable analyst reasons
  6. decision log    : one JSON line per decision (model version, rules, reasons, latency)
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.novelty.rules_engine import evaluate_rules_row, load_rules
from src.labels.policy import LabelPolicy
from src.serving.online_features import DuplicateTransactionError, OnlineFeatureEngine
from src.training.blending import CalibratedRankBlender

# Readable reason templates by feature family / prefix (first match wins)
REASONS = [
    ("_lbl_", "fraud previously confirmed on a linked {entity}"),
    ("other_clients_lbl", "other clients sharing this {entity} had confirmed fraud"),
    ("cluster_", "linked identity cluster is risky"),
    ("ring_", "device and card/address shared by several clients"),
    ("_clients_", "many clients on the same {entity} recently"),
    ("pair_is_new", "first time this {entity} is used by this client"),
    ("uid_new_", "first time this client uses this {entity}"),
    ("uid_changed_", "{entity} changed since the client's last transaction"),
    ("uid_n_", "unusual transaction velocity for this client"),
    ("_n_1h", "burst of transactions in the last hour"),
    ("ema_", "spending pattern deviates from the client's profile"),
    ("amt_z", "amount unusual for this client"),
    ("amt_ratio", "amount unusual for this client"),
    ("hmm_", "transaction sequence is unusual for this client"),
    ("hour_typicality", "unusual time of day for this client"),
    ("addr2_foreign", "billing country differs from the usual"),
    ("amt_gt2_decimals", "foreign-currency amount"),
    ("card3_not150", "foreign-issued card"),
    ("is_proxy", "connection through an anonymising proxy"),
    ("email_class", "risky email provider"),
    ("browser_version_lag", "outdated browser version"),
    ("count", "rare value for {entity}"),
]


def reason_text(feature: str) -> str:
    for key, tpl in REASONS:
        if key in feature:
            ent = feature.split("_lbl_")[0].split("_clients_")[0].replace("uid_new_", "").replace("uid_changed_", "")
            return tpl.format(entity=ent)
    return f"high-risk value of {feature}"


class ModelBundle:
    """Loads exactly what run_training.py wrote: bundle_<kind>.json + models + blender."""

    def __init__(self, art_dir: str | Path, kind: str = "refit"):
        self.dir = Path(art_dir)
        self.spec = json.loads((self.dir / f"bundle_{kind}.json").read_text())
        self.features: List[str] = self.spec["features"]
        self.cat_cols: List[str] = self.spec["cat_cols"]
        # models were trained on float32 frames: cast online features the same way so split decisions match
        self.dtype = np.dtype(self.spec.get("feature_dtype", "float32"))
        self.blender = CalibratedRankBlender.load(self.dir / self.spec["blender"])
        self.models: Dict[str, list] = {}
        for name, files in self.spec["model_files"].items():
            self.models[name] = [self._load(name, self.dir / f) for f in files]
        self.version = f"{self.spec['tag']}:{kind}:{int((self.dir / self.spec['blender']).stat().st_mtime)}"
        self.fast_model = True                       # numpy single-row path (identical scores); False = DataFrame path
        # a LightGBM model trained with pandas-categorical columns needs the DataFrame mapping
        self._lgb_numpy = all(getattr(m, "pandas_categorical", None) in (None, []) for m in self.models.get("lgbm", []))

    def fingerprint(self) -> str:
        """Identity of the model for snapshot validation: features, files, tag, label policy."""
        blob = {k: self.spec.get(k) for k in ("tag", "features", "cat_cols", "model_files", "blender", "label_policy")}
        return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def label_policy(self) -> Optional[LabelPolicy]:
        return LabelPolicy.from_cfg(self.spec["label_policy"]) if self.spec.get("label_policy") else None

    @staticmethod
    def _load(name, path):
        if name == "lgbm":
            import lightgbm as lgb
            return lgb.Booster(model_file=str(path))
        if name == "xgb":
            import xgboost as xgb
            m = xgb.XGBClassifier(); m.load_model(str(path))
            m.set_params(device="cpu"); m.get_booster().set_param({"device": "cpu"})   # serve on CPU
            return m
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(); m.load_model(str(path)); return m

    def _predict(self, name, model, X: pd.DataFrame) -> np.ndarray:
        X = X.astype(self.dtype)
        if name == "lgbm":
            return model.predict(X)
        if name == "catboost":
            X = X.copy()
            for c in self.cat_cols:
                X[c] = X[c].fillna(-1).astype(np.int64)
        return model.predict_proba(X)[:, 1]

    # ── single-row numpy API (no DataFrame): x = 1-D vector in ``self.features`` order ──────────────────
    def vector(self, vals: Dict[str, float]) -> np.ndarray:
        """float32 row (1, n_features) in this bundle's column order from {feature: float}."""
        return np.fromiter((vals[f] for f in self.features), dtype=np.float64,
                           count=len(self.features)).astype(self.dtype).reshape(1, -1)

    def _predict_array(self, name, model, A: np.ndarray) -> np.ndarray:
        if name == "lgbm" and self._lgb_numpy:
            return model.predict(A, num_threads=1)               # 1 row: threading only adds overhead
        return self._predict(name, model, pd.DataFrame(A, columns=self.features))

    def predict_proba_array(self, A: np.ndarray) -> np.ndarray:
        per_model = {n: np.mean([self._predict_array(n, m, A) for m in ms], axis=0) for n, ms in self.models.items()}
        return self.blender.predict_proba(per_model)

    def reasons_array(self, A: np.ndarray, top: int = 3) -> List[List[Dict]]:
        if "lgbm" not in self.models:
            return [[]]
        if not self._lgb_numpy:
            return self.reasons(pd.DataFrame(A, columns=self.features), top)
        contrib = self.models["lgbm"][0].predict(A, pred_contrib=True, num_threads=1)[:, :-1]
        out = []
        for i in range(len(A)):
            idx = np.argsort(-contrib[i])[:top]
            out.append([{"feature": self.features[j], "contribution": round(float(contrib[i, j]), 4),
                         "value": None if np.isnan(A[i, j]) else float(A[i, j]),
                         "reason": reason_text(self.features[j])} for j in idx if contrib[i, j] > 0])
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        per_model = {n: np.mean([self._predict(n, m, X) for m in ms], axis=0) for n, ms in self.models.items()}
        return self.blender.predict_proba(per_model)

    def reasons(self, X: pd.DataFrame, top: int = 3) -> List[List[Dict]]:
        """TreeSHAP (LightGBM pred_contrib) top positive contributions per row."""
        if "lgbm" not in self.models:
            return [[] for _ in range(len(X))]
        X = X.astype(self.dtype)
        contrib = self.models["lgbm"][0].predict(X, pred_contrib=True)[:, :-1]
        out = []
        for i in range(len(X)):
            idx = np.argsort(-contrib[i])[:top]
            out.append([{"feature": self.features[j], "contribution": round(float(contrib[i, j]), 4),
                         "value": None if pd.isna(X.iloc[i, j]) else float(X.iloc[i, j]),
                         "reason": reason_text(self.features[j])} for j in idx if contrib[i, j] > 0])
        return out


class CostPolicy:
    """Choose the action with the lowest expected cost (all costs in the amount currency)."""

    def __init__(self, cfg: Dict):
        self.c = cfg

    def expected_costs(self, p: float, amount: float) -> Dict[str, float]:
        c = self.c
        loss = amount * c["fraud_loss_rate"] + c["chargeback_fee"]          # cost if fraud is approved
        legit_value = amount * c["margin_rate"]                             # profit lost if legit is declined
        return {
            "approve": p * loss,
            "step_up": c["step_up_cost"] + p * loss * (1 - c["step_up_fraud_stop_rate"])
                       + (1 - p) * legit_value * c["step_up_legit_abandon_rate"],
            "review": c["review_cost"] + p * loss * (1 - c["review_fraud_stop_rate"]),
            # a wrongly declined good customer costs the sale AND future business (churn)
            "decline": (1 - p) * (legit_value + c.get("false_decline_churn_cost", 0.0)) + c["decline_cost"],
        }

    def decide(self, p: float, amount: float) -> Dict:
        costs = self.expected_costs(p, amount)
        if p < self.c.get("review_min_p", 0.0):          # review queue capacity (run_policy_simulation.py)
            costs.pop("review")
        action = min(costs, key=costs.get)
        return {"action": action, "expected_costs": {k: round(v, 4) for k, v in costs.items()}}


class DecisionLog:
    def __init__(self, path: Optional[str | Path]):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rec: Dict) -> None:
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")


class DecisionService:
    """Thread-safe, idempotent decision service.

    Concurrency: engine state is serialised by the engine lock (feature step only, ~1-3 ms);
    model inference, rules and policy run outside it, in parallel across request threads.
    Idempotency: the same TransactionID returns the cached decision (``idempotent_replay``) and
    never touches state; concurrent requests for one id wait for the first to finish.
    """

    def __init__(self, engine: OnlineFeatureEngine, bundle: ModelBundle, policy: CostPolicy,
                 rules: Optional[List[Dict]] = None, lists: Optional[Dict] = None,
                 log: Optional[DecisionLog] = None, challenger: Optional[ModelBundle] = None,
                 use_client_smoothing: bool = False, smoothing_cache_size: int = 500_000):
        self.engine, self.bundle, self.policy = engine, bundle, policy
        self.rules = rules or []
        self.lists = lists or {"allow": {}, "block": {}}       # {"block": {"device_fp": {...}, ...}}
        self.log = log or DecisionLog(None)
        self.challenger = challenger
        self.smooth = use_client_smoothing
        self._client_scores: "OrderedDict" = OrderedDict()
        self._smooth_cap = smoothing_cache_size
        self._smooth_lock = threading.Lock()
        self._inflight: Dict = {}
        self._inflight_lock = threading.Lock()
        for name, b in (("champion", bundle), ("challenger", challenger)):
            if b is None:
                continue
            missing = [f for f in b.features if f not in engine.feature_cols]
            if missing:
                raise ValueError(f"{name} bundle needs features the engine does not produce: {missing[:5]}")
            pol = b.label_policy() if hasattr(b, "label_policy") else None
            if pol is not None and pol.to_dict() != engine.policy.to_dict():
                raise ValueError(f"{name} bundle was trained under a different label policy than the serving "
                                 f"features ({pol.to_dict()} vs {engine.policy.to_dict()})")

    def _listed(self, kind: str, row: Dict) -> Optional[str]:
        for field, values in self.lists.get(kind, {}).items():
            v = row.get(field)
            if v is not None and str(v) in values:
                return f"{kind}list:{field}"
        return None

    def _smoothed(self, uid, p: float) -> float:
        with self._smooth_lock:                                # causal running mean per client
            s, n = self._client_scores.pop(uid, (0.0, 0))
            s, n = s + p, n + 1
            self._client_scores[uid] = (s, n)
            while len(self._client_scores) > self._smooth_cap:
                self._client_scores.popitem(last=False)
        return 0.5 * p + 0.5 * s / n

    def decide(self, txn: Dict) -> Dict:
        txid = txn.get("TransactionID")
        hit = self.engine.cached_decision(txid)
        if hit is not None:
            return {**hit, "idempotent_replay": True}
        with self._inflight_lock:
            ev = self._inflight.get(txid)
            owner = ev is None
            if owner:
                ev = self._inflight[txid] = threading.Event()
        if not owner:                                          # same id in flight: wait, return its decision
            ev.wait(timeout=30)
            hit = self.engine.cached_decision(txid)
            if hit is None:
                raise DuplicateTransactionError(
                    f"TransactionID {txid} is being scored / was scored without a cached decision")
            return {**hit, "idempotent_replay": True}
        try:
            rec = self._decide(txn)
            self.engine.cache_decision(txid, rec)
            return rec
        finally:
            with self._inflight_lock:
                self._inflight.pop(txid, None)
            ev.set()

    def _decide(self, txn: Dict) -> Dict:
        t0 = time.perf_counter()
        arr = hasattr(self.bundle, "predict_proba_array") and getattr(self.bundle, "fast_model", True)
        if arr:
            row, vals = self.engine.featurize_vals(txn)        # features use history BEFORE this txn (locked)
            t_feat = time.perf_counter()
            X = self.bundle.vector(vals)                       # float32 (1, n) straight in model column order
            p = float(self.bundle.predict_proba_array(X)[0])
        else:                                                  # DataFrame path (stub bundles, debugging)
            row, x = self.engine.featurize(txn)
            t_feat = time.perf_counter()
            X = x[self.bundle.features].to_frame().T           # each bundle casts to its training dtype
            p = float(self.bundle.predict_proba(X)[0])
        uid = row.get("uid1")
        if self.smooth and uid is not None:
            p = self._smoothed(uid, p)
        t_model = time.perf_counter()

        rec = {"transaction_id": txn.get("TransactionID"), "event_time": txn.get("TransactionDT"),
               "amount": float(txn["TransactionAmt"]), "model_version": self.bundle.version,
               "p_fraud": round(p, 6)}
        listed_allow, listed_block = self._listed("allow", row), self._listed("block", row)
        rules_out = (evaluate_rules_row(row, self.rules, now=float(txn.get("TransactionDT", 0)), score=p)
                     if self.rules else {"action": None, "fired": [], "shadow_fired": []})   # no 100-col DataFrame without rules
        if listed_allow:
            action, source = "approve", listed_allow
        elif listed_block:
            action, source = "decline", listed_block
        elif rules_out["action"] == "allow":
            action, source = "approve", "rule"
        elif rules_out["action"] in ("block", "review", "step_up"):
            action, source = {"block": "decline"}.get(rules_out["action"], rules_out["action"]), "rule"
        else:
            pol = self.policy.decide(p, float(txn["TransactionAmt"]))
            action, source = pol["action"], "policy"
            rec["expected_costs"] = pol["expected_costs"]
        t_rules = time.perf_counter()

        rec.update({"action": action, "decided_by": source, "rules_fired": rules_out["fired"],
                    "shadow_rules_fired": rules_out["shadow_fired"]})
        if action != "approve" or p >= self.policy.c.get("explain_min_p", 0.2):
            rec["reasons"] = (self.bundle.reasons_array(X) if arr else self.bundle.reasons(X))[0]
        if self.challenger is not None:                        # Phase 8: shadow model, logged only
            if arr and hasattr(self.challenger, "predict_proba_array"):
                pc = float(self.challenger.predict_proba_array(self.challenger.vector(vals))[0])
            else:
                if arr:
                    x = pd.Series(vals, dtype=np.float64)
                Xc = x[self.challenger.features].to_frame().T  # the challenger's OWN feature list
                pc = float(self.challenger.predict_proba(Xc)[0])
            rec["challenger"] = {"model_version": self.challenger.version, "p_fraud": round(pc, 6)}
        rec["latency_ms"] = {"features": round((t_feat - t0) * 1e3, 2), "model": round((t_model - t_feat) * 1e3, 2),
                             "rules_policy": round((t_rules - t_model) * 1e3, 2),
                             "total": round((time.perf_counter() - t0) * 1e3, 2)}
        self.log.write(rec)
        return rec

    def record_label(self, txn_id, is_fraud: float, known_at: Optional[int] = None) -> str:
        return self.engine.record_label(txn_id, is_fraud, known_at)

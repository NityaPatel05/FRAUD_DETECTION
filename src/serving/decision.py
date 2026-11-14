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

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.novelty.rules_engine import evaluate_rules_row, load_rules
from src.serving.online_features import OnlineFeatureEngine
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
        self.blender = CalibratedRankBlender.load(self.dir / self.spec["blender"])
        self.models: Dict[str, list] = {}
        for name, files in self.spec["model_files"].items():
            self.models[name] = [self._load(name, self.dir / f) for f in files]
        self.version = f"{self.spec['tag']}:{kind}:{int((self.dir / self.spec['blender']).stat().st_mtime)}"

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
        if name == "lgbm":
            return model.predict(X)
        if name == "catboost":
            X = X.copy()
            for c in self.cat_cols:
                X[c] = X[c].fillna(-1).astype(np.int64)
        return model.predict_proba(X)[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        per_model = {n: np.mean([self._predict(n, m, X) for m in ms], axis=0) for n, ms in self.models.items()}
        return self.blender.predict_proba(per_model)

    def reasons(self, X: pd.DataFrame, top: int = 3) -> List[List[Dict]]:
        """TreeSHAP (LightGBM pred_contrib) top positive contributions per row."""
        if "lgbm" not in self.models:
            return [[] for _ in range(len(X))]
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
    def __init__(self, engine: OnlineFeatureEngine, bundle: ModelBundle, policy: CostPolicy,
                 rules: Optional[List[Dict]] = None, lists: Optional[Dict] = None,
                 log: Optional[DecisionLog] = None, challenger: Optional[ModelBundle] = None,
                 use_client_smoothing: bool = False):
        self.engine, self.bundle, self.policy = engine, bundle, policy
        self.rules = rules or []
        self.lists = lists or {"allow": {}, "block": {}}       # {"block": {"device_fp": {...}, ...}}
        self.log = log or DecisionLog(None)
        self.challenger = challenger
        self.smooth = use_client_smoothing
        self._client_scores: Dict = {}

    def _listed(self, kind: str, row: Dict) -> Optional[str]:
        for field, values in self.lists.get(kind, {}).items():
            v = row.get(field)
            if v is not None and str(v) in values:
                return f"{kind}list:{field}"
        return None

    def decide(self, txn: Dict) -> Dict:
        t0 = time.perf_counter()
        rf = self.engine.row_frame(pd.DataFrame([txn]))
        row = rf.iloc[0].to_dict()
        st = self.engine._stateful(row)                       # features use history BEFORE this txn
        x = self.engine._encode(rf.iloc[0], st)
        t_feat = time.perf_counter()

        X = x[self.bundle.features].to_frame().T.astype(np.float64)
        p = float(self.bundle.predict_proba(X)[0])
        uid = row.get("uid1")
        if self.smooth and uid is not None:                    # causal running mean per client
            s, n = self._client_scores.get(uid, (0.0, 0))
            s, n = s + p, n + 1
            self._client_scores[uid] = (s, n)
            p = 0.5 * p + 0.5 * s / n
        t_model = time.perf_counter()

        rec = {"transaction_id": txn.get("TransactionID"), "event_time": txn.get("TransactionDT"),
               "amount": float(txn["TransactionAmt"]), "model_version": self.bundle.version,
               "p_fraud": round(p, 6)}
        listed_allow, listed_block = self._listed("allow", row), self._listed("block", row)
        rules_out = evaluate_rules_row(row, self.rules, now=float(txn.get("TransactionDT", 0)), score=p)
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
            rec["reasons"] = self.bundle.reasons(X)[0]
        if self.challenger is not None:                        # Phase 8: shadow model, logged only
            rec["challenger"] = {"model_version": self.challenger.version,
                                 "p_fraud": round(float(self.challenger.predict_proba(X)[0]), 6)}
        rec["latency_ms"] = {"features": round((t_feat - t0) * 1e3, 2), "model": round((t_model - t_feat) * 1e3, 2),
                             "rules_policy": round((t_rules - t_model) * 1e3, 2),
                             "total": round((time.perf_counter() - t0) * 1e3, 2)}
        self.log.write(rec)
        return rec

    def record_label(self, txn_id, is_fraud: float) -> None:
        self.engine.record_label(txn_id, is_fraud)

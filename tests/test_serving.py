"""Phase 7 decision layer — synthetic data and a stub model (no training)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.serving.decision import CostPolicy, DecisionLog, DecisionService, reason_text

COSTS = {"fraud_loss_rate": 1.0, "chargeback_fee": 15.0, "margin_rate": 0.03, "decline_cost": 0.5, "false_decline_churn_cost": 20.0,
         "step_up_cost": 0.3, "step_up_fraud_stop_rate": 0.8, "step_up_legit_abandon_rate": 0.05,
         "review_cost": 3.0, "review_fraud_stop_rate": 0.9, "review_min_p": 0.0, "explain_min_p": 0.2}


@pytest.fixture(scope="module")
def world():
    from test_v2_pipeline import _synthetic_ieee
    from src.feature_engineering.pipeline_v2 import run_feature_pipeline_v2
    full = _synthetic_ieee(2000, 5)
    tr = full[full.TransactionDT < 86400 * 60]; va = full[full.TransactionDT >= 86400 * 60]
    te = _synthetic_ieee(300, 6, start_id=50_000, labelled=False); te["TransactionDT"] += 86400 * 90
    data, meta = run_feature_pipeline_v2(tr, va, te, {"label_delay_days": 10})
    return meta, tr, va, te


class StubBundle:
    def __init__(self, features, p):
        self.features, self.p, self.version, self.spec = features, p, "stub:1", {}

    def predict_proba(self, X):
        assert list(X.columns) == self.features
        return np.array([self.p])

    def reasons(self, X, top=3):
        return [[{"feature": "uid1_lbl_rate", "reason": reason_text("uid1_lbl_rate")}]]


def _service(meta, tr, va, p, **kw):
    from src.serving.online_features import OnlineFeatureEngine
    eng = OnlineFeatureEngine(meta)
    eng.replay(pd.concat([tr, va]), labels=True, collect=False)
    return DecisionService(eng, StubBundle(meta["feature_cols"][:50], p), CostPolicy(COSTS), **kw)


def test_cost_policy_actions():
    pol = CostPolicy(COSTS)
    assert pol.decide(0.001, 50)["action"] == "approve"
    assert pol.decide(0.95, 500)["action"] == "decline"
    mid = pol.decide(0.2, 300)["action"]
    assert mid in ("step_up", "review")
    assert CostPolicy({**COSTS, "review_min_p": 0.9}).decide(0.2, 300)["action"] != "review"
    c = pol.expected_costs(0.5, 100)
    assert c["approve"] == pytest.approx(0.5 * 115)


def test_decision_service_layers(world, tmp_path):
    meta, tr, va, te = world
    log = tmp_path / "d.jsonl"
    txn = te.drop(columns=[c for c in te.columns if c == "isFraud"]).iloc[0].to_dict()
    svc = _service(meta, tr, va, p=0.001, log=DecisionLog(log))
    out = svc.decide(txn)
    assert out["action"] == "approve" and out["decided_by"] == "policy"
    assert set(out["latency_ms"]) == {"features", "model", "rules_policy", "total"}

    hi = _service(meta, tr, va, p=0.99)
    big = {**txn, "TransactionAmt": 900.0}
    assert hi.decide(big)["action"] == "decline" and "reasons" in hi.decide(big)

    # block list beats the model; enforced allow rule beats block rules; shadow only logged
    blk = _service(meta, tr, va, p=0.001, lists={"block": {"ProductCD": {str(txn["ProductCD"])}}})
    assert blk.decide(txn)["action"] == "decline"
    rules = [{"id": "allow_small", "conditions": [{"col": "TransactionAmt", "op": "lt", "value": 1e9}],
              "action": "allow", "mode": "enforce", "expires_at": None},
             {"id": "block_all", "conditions": [{"col": "TransactionAmt", "op": "gt", "value": 0}],
              "action": "block", "mode": "enforce", "expires_at": None},
             {"id": "shadow1", "conditions": [{"col": "TransactionAmt", "op": "gt", "value": 0}],
              "action": "block", "mode": "shadow", "expires_at": None}]
    r = _service(meta, tr, va, p=0.99, rules=rules).decide(txn)
    assert r["action"] == "approve" and r["decided_by"] == "rule" and r["shadow_rules_fired"] == ["shadow1"]
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert recs and recs[0]["model_version"] == "stub:1"


def test_engine_state_snapshot_roundtrip(world, tmp_path):
    from src.serving.online_features import OnlineFeatureEngine
    meta, tr, va, te = world
    a = OnlineFeatureEngine(meta); a.replay(pd.concat([tr, va]), labels=True, collect=False)
    a.save_state(tmp_path / "s.pkl")
    b = OnlineFeatureEngine(meta); b.load_state(tmp_path / "s.pkl")
    txn = te.iloc[3].to_dict()
    pd.testing.assert_series_equal(a.transform(txn), b.transform(txn))


def test_realised_cost_accounting():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ps", "scripts/run_policy_simulation.py")
    ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)
    act = np.array(["approve", "approve", "decline", "decline", "review"])
    y = np.array([1, 0, 1, 0, 1]); amt = np.array([100.0, 100, 100, 100, 100])
    c = ps.realised_cost(act, y, amt, COSTS)
    np.testing.assert_allclose(c, [115, 0, 0.5, 3 + 20 + 0.5, 3 + 115 * 0.1])

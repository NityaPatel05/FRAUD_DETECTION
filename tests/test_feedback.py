"""Phase 8 feedback loop — synthetic data only, no training."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feedback.labels import LabelStore
from src.feedback.promotion import promotion_gate, quantile_map
from src.feedback.exploration import allow_probability, ips_policy_metrics, naive_policy_metrics
from src.feedback.monitoring import daily_report, alerts
from src.jobs.retrain_v2 import ModelRegistry

D = 86400


def test_label_maturity_rules():
    ls = LabelStore()
    ls.add(1, 0 * D, 40 * D, 1, "chargeback")      # fraud, known at day 40
    ls.add(2, 0 * D, 1 * D, 0, "analyst")          # analyst said legit
    ls.add(2, 0 * D, 50 * D, 1, "chargeback")      # …but a chargeback arrived later
    txns = pd.DataFrame({"TransactionID": [1, 2, 3, 4], "TransactionDT": [0, 0, 0, 80 * D]})
    y30 = ls.training_labels(txns, as_of=30 * D, maturity_days=20)
    assert y30[0] == 0                                      # fraud not known yet → mature negative (as known then)
    y60 = ls.training_labels(txns, as_of=60 * D, maturity_days=20)
    assert y60.tolist()[:3] == [1.0, 1.0, 0.0]               # chargeback overrides analyst; txn 3 mature legit
    assert np.isnan(y60[3])                                  # future transaction
    y10 = ls.training_labels(txns, as_of=10 * D, maturity_days=20)
    assert np.isnan(y10[2])                                  # immature → excluded, NOT legit


def test_promotion_gate_accepts_better_and_rejects_queue_flood():
    r = np.random.default_rng(0)
    y = (r.random(20000) < 0.04).astype(float)
    champ = np.clip(0.3 * y + r.normal(0.2, 0.15, len(y)), 0, 1)
    chall = np.clip(0.5 * y + r.normal(0.2, 0.15, len(y)), 0, 1)
    act = lambda p, t: np.where(p > t, "decline", "approve")   # noqa: E731
    ok = promotion_gate(y, champ, chall, act(champ, 0.6), act(quantile_map(chall, champ)(chall), 0.6))
    assert ok["promote"], ok["reasons"]
    flood = promotion_gate(y, champ, chall, act(champ, 0.6), act(chall, 0.3))   # thresholds not mapped
    assert not flood["promote"] and any("SafetyNet" in m for m in flood["reasons"])
    worse = promotion_gate(y, chall, champ, act(chall, 0.6), act(champ, 0.6))
    assert not worse["promote"]


def test_ips_is_unbiased_where_naive_is_not():
    """Unbiasedness = correct on AVERAGE over exploration draws (one draw with ~tens of
    explored rows is noisy by nature); naive estimates stay biased however often you draw."""
    r = np.random.default_rng(1)
    n = 50_000
    y = (r.random(n) < 0.03).astype(float)
    p = np.clip(0.6 * y + r.normal(0.1, 0.12, n), 0, 1)
    block = p >= np.quantile(p, 0.99)
    prop = allow_probability(p, block, 0.05, 0.01)
    cand = p >= np.quantile(p, 0.98)
    truth = (y * cand).sum() / y.sum()
    ips, naive = [], []
    for seed in range(200):
        obs = np.random.default_rng(seed).random(n) < prop
        ips.append(ips_policy_metrics(y, obs, prop, cand)["recall"])
        naive.append(naive_policy_metrics(y, obs, cand)["recall"])
    assert abs(np.mean(ips) - truth) < 0.02
    assert abs(np.mean(naive) - truth) > 0.05               # naive misses the blocked fraud


def test_monitoring_alerts():
    days = np.repeat(np.arange(20), 2000)
    log = pd.DataFrame({"day": days, "event_time": days * D, "transaction_id": np.arange(len(days)),
                        "p_fraud": np.r_[np.random.default_rng(2).random(30000) * 0.1,
                                         np.random.default_rng(3).random(10000) * 0.9],
                        "action": np.where(days >= 15, "decline", "approve"),
                        "latency_ms.total": 10.0})
    rep = daily_report(log, ref_days=7)
    al = alerts(rep, {"rate_decline": [0, 0.03], "score_psi": [0, 0.2], "p99_latency_ms": [0, 50]})
    assert set(al["metric"]) == {"rate_decline", "score_psi"} and al["day"].min() == 15


def test_registry_lifecycle(tmp_path):
    reg = ModelRegistry(tmp_path / "r.json")
    reg.register_challenger("artifacts/v2/causal", {"auc_pr": 0.7})
    reg.promote_challenger({"promote": True})
    reg2 = ModelRegistry(tmp_path / "r.json")
    assert reg2.data["champion"]["art_dir"] == "artifacts/v2/causal" and reg2.data["challenger"] is None
    with pytest.raises(ValueError):
        reg2.promote_challenger()

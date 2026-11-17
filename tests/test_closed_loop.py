"""Phase C2 closed-loop experiments: no-truth learning, IPW, expected-cost actions, drift trigger."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.closed_loop import actions, features, policy as policy_mod, store as store_mod, triggers
from src.closed_loop.actions import CostRule, SegmentThresholdRule, choose_actions, expected_costs, tune_threshold
from src.closed_loop.config import COST_DEFAULTS, MODEL_QUICK, make_world_cfg
from src.closed_loop.policy import ModelPolicy
from src.closed_loop.store import Chunk, OutcomeStore
from src.closed_loop.triggers import CalendarTrigger, DriftTrigger
from src.closed_loop.weights import ipw_weights
from src.simulator import simulate


# ───────────────────────── never reads ground truth ─────────────────────────
def test_policy_sources_never_mention_ground_truth():
    for mod in (policy_mod, store_mod, features, actions, triggers):
        src = open(mod.__file__, encoding="utf-8").read()
        for bad in ("is_fraud", "fraud_scenario", "incident_id", "gang_id", "ring_id"):
            assert bad not in src, f"{mod.__name__} mentions {bad}"


class _Poisoned(ModelPolicy):
    """Same policy, but the frame it is shown carries a random truth-looking column (a leak attempt)."""

    def decide(self, batch):
        rng = np.random.default_rng(batch.day)
        batch.frame["is_fraud_true"] = rng.integers(0, 2, len(batch.frame))
        batch.frame["fraud_scenario"] = "legit"
        return super().decide(batch)


def _mk(cls):
    params = dict(MODEL_QUICK, n_estimators=15, n_jobs=1)
    return cls("t", CostRule(), CalendarTrigger(10), warmup_days=30, gap_days=15, model_params=params, seed=3, min_pos=5)


@pytest.fixture(scope="module")
def two_runs():
    def cfg():
        return make_world_cfg(5, "quick", n_customers=700, n_merchants=120, n_days=60, val_days=3, test_days=8,
                              n_gangs=4, explore_rate=0.02)
    a, b = _mk(ModelPolicy), _mk(_Poisoned)
    ra, rb = simulate(cfg(), policy=a), simulate(cfg(), policy=b)
    return a, b, ra.transactions, rb.transactions


def test_flipping_truth_columns_changes_no_decision(two_runs):
    a, b, ta, tb = two_runs
    assert len(ta) == len(tb)
    assert (ta["decision"].to_numpy() == tb["decision"].to_numpy()).all()
    assert np.allclose(ta["risk_score"].astype(float), tb["risk_score"].astype(float), equal_nan=True)
    assert len(a.retrains) >= 1 and len(a.retrains) == len(b.retrains)       # it did learn, from delivered labels only


def test_learning_policy_uses_the_model_after_warmup(two_runs):
    a, _, ta, _ = two_runs
    assert a.model is not None and a.snapshots
    assert set(ta["decision"].unique()) <= set(actions.ACTION_ORDER)


# ───────────────────────── IPW ─────────────────────────
def test_ipw_weights_are_inverse_propensity():
    w = ipw_weights(np.array([0.02, 1.0, 0.05, np.nan, 0.0]), max_weight=1000)
    assert np.allclose(w, [50, 1, 20, 1, 1])
    assert np.allclose(ipw_weights(np.array([0.001]), max_weight=100), [100])          # clipped


def test_training_set_weights_use_logged_propensity():
    st = OutcomeStore()
    n = 6
    X = np.zeros((n, 2), np.float32)
    c = Chunk(0, np.arange(1, n + 1), X, np.zeros(n), np.array(["CNP|est"] * n), np.full(n, 10.0))
    c.approved[:] = [1, 1, 1, 0, 1, 1]
    c.prop[:] = [1, 0.02, 1, 0.98, 0.05, 1]
    st.chunks.append(c)
    st.rep = {2: 1.0, 5: 1.0}
    _, y, w, _, _, tid = st.training_set(cutoff_day=100, gap_days=10)
    assert list(tid) == [1, 2, 3, 5, 6]                       # the declined row (id 4) has no outcome and is excluded
    assert np.allclose(w, [1, 50, 1, 20, 1])                  # explored rows: 1/propensity
    assert list(y) == [0, 1, 0, 1, 0]
    _, _, w3, *_ = st.training_set(cutoff_day=100, gap_days=10, use_ipw=False)
    assert np.allclose(w3, 1)
    assert st.training_set(cutoff_day=5, gap_days=10) is None  # rows newer than the label-maturity gap are not used


# ───────────────────────── expected-cost action selection ─────────────────────────
def _argmin_reference(p, amt, step_ok, c=COST_DEFAULTS):
    loss = c["fraud_loss_frac"] * amt + c["chargeback_fee"]
    fd = c["margin_rate"] * amt + c["churn_cost"]
    step = (p * c["stepup_fraud_pass"] * (1 - c["liability_shift_share"]) * loss
            + (1 - p) * (1 - c["stepup_legit_pass"]) * c["margin_rate"] * amt + c["stepup_fee"]) if step_ok else np.inf
    review = (c["review_cost"] + p * (1 - c["review_accuracy"]) * loss
              + (1 - p) * (c["review_false_decline"] * fd + c["review_abandon"] * c["margin_rate"] * amt))
    return {"approve": p * loss, "decline": (1 - p) * fd, "step_up_3ds": step, "review": review}


@pytest.mark.parametrize("p,amt,step_ok,cap,want", [
    (0.001, 50.0, True, 5, "approve"),         # almost surely legit
    (0.90, 400.0, True, 0, "decline"),         # almost surely fraud, no review capacity
    (0.10, 200.0, True, 5, "step_up_3ds"),     # in doubt on a CNP row: 3DS is the cheapest answer
    (0.10, 200.0, False, 5, "review"),         # same row in a channel without 3DS: review
    (0.10, 200.0, False, 0, "approve"),        # ... and with no analyst capacity: approve
])
def test_expected_cost_picks_the_argmin(p, amt, step_ok, cap, want):
    costs = expected_costs(np.array([p]), np.array([amt]), np.array([step_ok]))
    got = choose_actions(costs, cap)[0]
    ref = _argmin_reference(p, amt, step_ok)
    if cap == 0:
        ref.pop("review")
    assert got == want == min(ref, key=ref.get)


def test_review_capacity_goes_to_largest_saving():
    p = np.array([0.10, 0.10, 0.10])
    amt = np.array([50.0, 900.0, 300.0])
    costs = expected_costs(p, amt, np.zeros(3, bool))
    act = choose_actions(costs, 1)
    assert list(act).count("review") == 1 and act[1] == "review"     # the biggest ticket is reviewed first


def test_cost_rule_on_a_frame_respects_step_up_availability():
    f = pd.DataFrame({"amount_usd": [200.0, 200.0], "channel": ["CNP", "CP"], "three_ds_result": ["not_used", "not_used"]})
    act = CostRule({"review_capacity_frac": 0.0}).decide(f, np.array([0.10, 0.10]))
    assert act[0] == "step_up_3ds" and act[1] == "approve"


def test_segment_thresholds_are_tuned_per_segment():
    rng = np.random.default_rng(0)
    n = 6000
    seg = np.array(["CNP|new"] * (n // 2) + ["CP|est"] * (n // 2))
    y = (rng.random(n) < np.where(seg == "CNP|new", 0.08, 0.004)).astype(int)
    p = np.clip(0.02 + 0.5 * y * rng.random(n) + 0.05 * rng.random(n), 0, 1)
    amt = np.full(n, 100.0)
    rule = SegmentThresholdRule({}, 0.3)
    rule.fit(p, y, np.ones(n), amt, seg, segmented=True)
    assert set(rule.thresholds) == {"CNP|new", "CP|est"}
    assert 0.02 <= tune_threshold(p, y, np.ones(n), amt) <= 0.98


# ───────────────────────── drift trigger ─────────────────────────
class _FakeMon:
    def __init__(self, shift_from):
        self.shift_from, self.rng = shift_from, np.random.default_rng(0)

    def _x(self, d0, d1, n=400):
        shifted = d0 >= self.shift_from
        return self.rng.normal(2.5 if shifted else 0.0, 1.0, n * max(d1 - d0, 1))

    def scores(self, d0, d1):
        return 1 / (1 + np.exp(-self._x(d0, d1)))

    def feature(self, name, d0, d1):
        return self._x(d0, d1)

    def matured_perf(self, d0, d1):
        return 0.5, 20


def test_drift_trigger_fires_on_shift_and_not_on_stationary_stream():
    kw = dict(features=["f1", "f2", "f3", "f4"], check_every=7, window_days=7, ref_days=14, min_gap_days=14)
    tr, tr2 = DriftTrigger(**kw), DriftTrigger(**kw)
    fired_stat = [tr.should_retrain(d, 0, _FakeMon(10 ** 6))[0] for d in range(0, 120)]
    shifted = _FakeMon(40)
    fired_shift = [tr2.should_retrain(d, 0, shifted)[0] for d in range(0, 120)]
    assert not any(fired_stat)
    assert any(fired_shift)
    assert fired_shift.index(True) > 40                       # not before the shift happened


def test_drift_evaluate_flags_performance_drop_with_enough_positives():
    tr = DriftTrigger(perf_drop=0.3, perf_min_pos=10)
    z = np.random.default_rng(1).normal(size=500)
    ok = tr.evaluate(z, z, {}, {}, (0.5, 30), (0.45, 30))
    bad = tr.evaluate(z, z, {}, {}, (0.5, 30), (0.2, 30))
    few = tr.evaluate(z, z, {}, {}, (0.5, 30), (0.2, 3))
    assert not ok["fire"] and bad["fire"] and not few["fire"]


def test_calendar_trigger():
    t = CalendarTrigger(30)
    assert not t.should_retrain(29, 0, None)[0] and t.should_retrain(30, 0, None)[0]

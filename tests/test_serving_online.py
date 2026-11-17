"""Phase A2 -- production-safe online serving: label policy parity, idempotency, threads, pruning,
snapshot fingerprint, challenger features, unknown labels. Synthetic data, stub models (no training)."""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.labels.policy import LabelPolicy, fraud_known_at, maturity_at
from src.serving.decision import CostPolicy, DecisionService
from src.serving.online_features import (
    DuplicateTransactionError, MissingFieldsError, OnlineFeatureEngine, SnapshotMismatchError,
)

# lognormal chargeback delay + maturity shorter than the synthetic horizon, so every counter moves
POLICY = {"delay_median_days": 8, "delay_sigma": 0.6, "delay_min_days": 1, "delay_cap_days": 30,
          "maturity_days": 20, "train_gap_days": 3}
COSTS = {"fraud_loss_rate": 1.0, "chargeback_fee": 15.0, "margin_rate": 0.03, "decline_cost": 0.5,
         "false_decline_churn_cost": 20.0, "step_up_cost": 0.3, "step_up_fraud_stop_rate": 0.8,
         "step_up_legit_abandon_rate": 0.05, "review_cost": 3.0, "review_fraud_stop_rate": 0.9,
         "review_min_p": 0.0, "explain_min_p": 0.2}


@pytest.fixture(scope="module")
def world():
    from test_v2_pipeline import _synthetic_ieee
    from src.feature_engineering.pipeline_v2 import run_feature_pipeline_v2
    full = _synthetic_ieee(3000, 11)
    tr = full[full.TransactionDT < 86400 * 60]
    va = full[full.TransactionDT >= 86400 * 60]
    te = _synthetic_ieee(600, 12, start_id=50_000, labelled=False)
    te["TransactionDT"] += 86400 * 90
    data, meta = run_feature_pipeline_v2(tr, va, te, {"label_policy": POLICY})
    return data, meta, (tr, va, te)


class StubBundle:
    def __init__(self, features, p=0.001, tag="stub"):
        self.features, self.p, self.version, self.spec = features, p, f"{tag}:1", {}
        self.seen_cols = []

    def predict_proba(self, X):
        assert list(X.columns) == self.features, "bundle must be scored on ITS OWN feature list"
        self.seen_cols.append(list(X.columns))
        return np.array([self.p])

    def reasons(self, X, top=3):
        return [[]]


def _svc(meta, **kw):
    eng = OnlineFeatureEngine(meta, kw.pop("serving", None))
    return DecisionService(eng, StubBundle(meta["feature_cols"][:40]), CostPolicy(COSTS), **kw)


def _txn(df, i):
    d = df.iloc[i].to_dict()
    d.pop("isFraud", None)
    return d


# ── 1. honest parity: only frauds are posted, negatives become mature by themselves ────────────

def test_parity_fraud_only_labels_and_scores(world):
    data, meta, (tr, va, te) = world
    off = (pd.concat([data["train"], data["val"], data["test"]], ignore_index=True)
             .sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True))
    raw = pd.concat([tr, va, te], ignore_index=True)
    eng = OnlineFeatureEngine(meta)
    assert eng.policy.to_dict() == LabelPolicy.from_cfg(POLICY).to_dict()
    on = eng.replay(raw, labels=True)                       # only isFraud==1 rows are posted as labels
    feats = meta["feature_cols"]
    a = off[feats].to_numpy(dtype=np.float64)
    b = on[feats].to_numpy(dtype=np.float64)
    bad = []
    for j, c in enumerate(feats):
        ok = np.isclose(a[:, j], b[:, j], rtol=1e-5, atol=1e-5, equal_nan=True)
        if not ok.all():
            k = np.where(~ok)[0]
            bad.append(f"{c}: {len(k)} rows differ, e.g. row {k[0]} offline={a[k[0], j]} online={b[k[0], j]}")
    assert not bad, "online/offline skew:\n" + "\n".join(bad[:25])
    # the label features must actually carry signal here (otherwise the test proves nothing)
    n, f = on["uid1_lbl_n"], on["uid1_lbl_fraud"]
    assert n.max() >= 2 and f.max() >= 1
    assert (n > f).any(), "mature negatives must be counted although no negative label was ever posted"
    assert on["device_fp_lbl_n"].max() >= 1 and on["cluster_lbl_fraud"].max() >= 1

    # scores: a model fitted offline scores online features identically (float32, same splits)
    import lightgbm as lgb
    tr_off = off[off["TransactionDT"] < 86400 * 60]
    y = tr_off["isFraud"].fillna(0).to_numpy()
    m = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 15, "seed": 1, "deterministic": True,
                   "min_data_in_leaf": 10}, lgb.Dataset(tr_off[feats].astype(np.float32), y), 30)
    p_off = m.predict(off[feats].astype(np.float32))
    p_on = m.predict(on[feats].astype(np.float32))
    np.testing.assert_allclose(p_on, p_off, atol=1e-6)


def test_fraud_label_becomes_visible_at_policy_time(world):
    data, meta, (tr, va, te) = world
    eng = OnlineFeatureEngine(meta)
    fr = tr[tr.isFraud == 1].iloc[0]
    sub = tr[tr.TransactionDT <= fr.TransactionDT]
    eng.replay(sub, labels=True, collect=False)
    t, keys, uid = eng.txn_keys[fr.TransactionID]
    ns = "lh_uid1"
    k = keys[ns]
    fk = int(fraud_known_at(fr.TransactionID, t, eng.policy))
    mat = int(maturity_at(t, eng.policy))
    own = lambda: (eng.delayed[(ns, k)].n, eng.delayed[(ns, k)].f, eng.delayed[(ns, k)].fm)   # noqa: E731
    base_n = None
    eng._apply_events(min(fk, mat) - 1)
    n0, f0, fm0 = own()
    eng._apply_events(fk)
    n1, f1, fm1 = own()
    assert f1 == f0 + 1 and (fk >= mat or fm1 == fm0)          # fraud visible exactly at t + delay
    eng._apply_events(max(fk, mat))
    n2, f2, fm2 = own()
    assert fm2 == fm0 + 1 and n2 >= n1                         # fraud-and-mature at max(fk, maturity)
    eng._apply_events(mat)
    assert eng.delayed[(ns, k)].n >= 1


# ── 2. idempotency ───────────────────────────────────────────────────────────────────────────

def test_idempotent_rescore_does_not_mutate_state(world):
    meta = world[1]; te = world[2][2]
    svc = _svc(meta)
    txn = _txn(te, 0)
    r1 = svc.decide(txn)
    uid = svc.engine.txn_keys[txn["TransactionID"]][2]
    cnt = svc.engine.count[("uid", uid)]
    n_scored, ev = svc.engine.n_scored, len(svc.engine.events)
    r2 = svc.decide(txn)
    assert r2["idempotent_replay"] and r2["p_fraud"] == r1["p_fraud"] and r2["action"] == r1["action"]
    assert svc.engine.count[("uid", uid)] == cnt and svc.engine.n_scored == n_scored
    assert len(svc.engine.events) == ev
    with pytest.raises(DuplicateTransactionError):
        svc.engine.featurize(txn)                           # engine level: refuses, no mutation
    assert svc.engine.n_scored == n_scored


# ── 3. threads ───────────────────────────────────────────────────────────────────────────────

def test_concurrent_scoring_is_consistent(world):
    meta = world[1]; te = world[2][2]
    txns = [_txn(te, i) for i in range(150)]
    seq = _svc(meta)
    for t in txns:
        seq.decide(t)
    par = _svc(meta)
    errs = []

    def work(t):
        try:
            return par.decide(t)
        except Exception as e:                              # noqa: BLE001
            errs.append(e)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(work, txns + txns[:40]))               # 40 duplicates racing the originals
    assert not errs
    assert par.engine.n_scored == len(txns) == len(par.engine.txn_keys)
    for key in {k for k in seq.engine.count if k[0] == "uid"}:   # order-independent aggregate
        assert par.engine.count[key] == seq.engine.count[key]
    # same id from many threads at once -> one decision
    one = _txn(te, 200)
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda _: par.decide(one), range(16)))
    assert len({o["p_fraud"] for o in outs}) == 1 and par.engine.n_scored == len(txns) + 1


# ── 4. bounded state ─────────────────────────────────────────────────────────────────────────

def test_pruning_keeps_in_window_features_identical(world):
    data, meta, (tr, va, te) = world
    raw = pd.concat([tr, va, te], ignore_index=True)
    base = OnlineFeatureEngine(meta, {"prune_every": 0, "profile_ttl_days": None})
    on0 = base.replay(raw)
    # windows pruned often + idle profiles dropped (ttl is clamped to >= the longest window)
    pr = OnlineFeatureEngine(meta, {"prune_every": 100, "profile_ttl_days": 35})
    on1 = pr.replay(raw)
    import re
    pat = re.compile(r"^(uid|device_fp|card1_addr1)_(n|amt_sum)_(1h|1d|7d|30d)$|^(device_fp|card1_addr1)_clients_\d+d$"
                     r"|_count30d$|^uid_n_10m$")
    win_cols = [c for c in meta["feature_cols"] if pat.search(c)]
    assert len(win_cols) >= 10
    pd.testing.assert_frame_equal(on0[win_cols], on1[win_cols], check_exact=False, rtol=1e-9, atol=1e-9)
    # pruning alone (no profile ttl) changes nothing at all
    only = OnlineFeatureEngine(meta, {"prune_every": 100, "profile_ttl_days": None})
    pd.testing.assert_frame_equal(on0, only.replay(raw), check_exact=False, rtol=1e-9, atol=1e-9)
    s0, s1 = base.state_size(), pr.prune()
    assert s1["after"] < sum(s0.values())                   # state actually shrank
    assert pr.state_size()["win"] <= s0["win"]


def test_txn_index_and_decision_cache_are_bounded(world):
    meta = world[1]; te = world[2][2]
    svc = _svc(meta, serving={"prune_every": 0, "decision_cache_size": 50, "label_grace_days": 0})
    for i in range(120):
        svc.decide(_txn(te, i))
    svc.engine.prune()
    assert len(svc.engine.decisions) == 50
    far = _txn(te, 121); far["TransactionDT"] = int(te.TransactionDT.max()) + 86400 * 400
    svc.decide(far)
    svc.engine.prune()                                      # everything older than delay cap/maturity is gone
    assert len(svc.engine.txn_keys) == 1 and len(svc.engine.index_order) == 1


# ── 5. snapshot fingerprint ──────────────────────────────────────────────────────────────────

def test_snapshot_fingerprint_refuses_mismatch(world, tmp_path):
    meta = world[1]; te = world[2][2]
    a = OnlineFeatureEngine(meta)
    a.featurize(_txn(te, 0))
    path = tmp_path / "s.pkl"
    a.save_state(path, extra={"bundle": "abc"})
    OnlineFeatureEngine(meta).load_state(path, expect_extra={"bundle": "abc"})       # same -> ok
    b = OnlineFeatureEngine(meta)
    with pytest.raises(SnapshotMismatchError, match="extra.bundle"):
        b.load_state(path, expect_extra={"bundle": "OTHER"})
    other_policy = {**meta, "label_policy": {**POLICY, "maturity_days": 45}}
    with pytest.raises(SnapshotMismatchError, match="label_policy"):
        OnlineFeatureEngine(other_policy).load_state(path)
    other_feats = {**meta, "feature_cols": meta["feature_cols"][:-1]}
    with pytest.raises(SnapshotMismatchError, match="feature_cols"):
        OnlineFeatureEngine(other_feats).load_state(path)
    import pickle
    legacy = tmp_path / "old.pkl"
    legacy.write_bytes(pickle.dumps({"win": {}}))
    with pytest.raises(SnapshotMismatchError, match="format"):
        OnlineFeatureEngine(meta).load_state(legacy)


def test_snapshot_roundtrip_keeps_labels_decisions_and_index(world, tmp_path):
    meta = world[1]; te = world[2][2]
    svc = _svc(meta)
    txn = _txn(te, 5)
    r = svc.decide(txn)
    svc.engine.save_state(tmp_path / "s.pkl")
    new = _svc(meta)
    new.engine.load_state(tmp_path / "s.pkl")
    assert new.decide(txn)["idempotent_replay"]                      # decision survived the restart
    assert new.record_label(txn["TransactionID"], 1) == "registered"  # label for a pre-restart txn
    assert r["transaction_id"] == txn["TransactionID"]


# ── 6. challenger / bundles ──────────────────────────────────────────────────────────────────

def test_challenger_scored_on_its_own_features(world):
    meta = world[1]; te = world[2][2]
    f = meta["feature_cols"]
    champ, chall = StubBundle(f[:40], 0.01, "champ"), StubBundle(f[20:90], 0.02, "chall")
    svc = DecisionService(OnlineFeatureEngine(meta), champ, CostPolicy(COSTS), challenger=chall)
    out = svc.decide(_txn(te, 1))
    assert out["challenger"]["p_fraud"] == 0.02 and chall.seen_cols == [f[20:90]]
    bad = StubBundle(f[:5] + ["not_a_feature"])
    with pytest.raises(ValueError, match="not_a_feature"):
        DecisionService(OnlineFeatureEngine(meta), champ, CostPolicy(COSTS), challenger=bad)


def test_bundle_policy_mismatch_refused(world):
    meta = world[1]

    class B(StubBundle):
        def label_policy(self):
            return LabelPolicy.from_cfg({**POLICY, "maturity_days": 99})
    with pytest.raises(ValueError, match="label policy"):
        DecisionService(OnlineFeatureEngine(meta), B(meta["feature_cols"][:5]), CostPolicy(COSTS))


# ── 7. labels ────────────────────────────────────────────────────────────────────────────────

def test_label_statuses_and_unknown_transaction(world):
    meta = world[1]; te = world[2][2]
    svc = _svc(meta)
    txn = _txn(te, 2)
    assert svc.record_label(999_999_999, 1) == "unknown_transaction"
    assert svc.engine.label_stats["unknown_transaction"] == 1
    svc.decide(txn)
    tid = txn["TransactionID"]
    assert svc.record_label(tid, float("nan")) == "invalid"
    assert svc.record_label(tid, 0) == "negative_ignored"
    n_ev = len(svc.engine.events)
    assert svc.record_label(tid, 1) == "registered" and len(svc.engine.events) == n_ev + 2
    assert svc.record_label(tid, 1) == "already_registered" and len(svc.engine.events) == n_ev + 2


# ── 8. full-schema bundle needs Vesta fields ────────────────────────────────────────────────

def test_full_schema_requires_vendor_fields(world):
    meta = world[1]; te = world[2][2]
    assert meta["schema"] == "full"
    eng = OnlineFeatureEngine(meta)
    gateway_only = {k: v for k, v in _txn(te, 3).items()
                    if not (k[0] in "VC" and k[1:].isdigit()) and not (k[0] == "D" and k[1:].isdigit() and k != "D1")}
    with pytest.raises(MissingFieldsError, match="Vesta"):
        eng.featurize(gateway_only)
    assert eng.n_scored == 0
    eng.featurize(_txn(te, 3))                                # complete payload is fine


def test_build_service_refuses_full_schema_for_live_serving():
    from src.serving.service_factory import build_service
    with pytest.raises(ValueError, match="Vesta"):
        build_service(schema="full")

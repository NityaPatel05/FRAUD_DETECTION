"""Sim track v2 on a tiny generated world (SimConfig.small): causality, label semantics, metrics."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim_track.evaluate import pct_vs_reference, sim_report
from src.sim_track.features import (BURST_PREFIX, GROUND_TRUTH, OUTCOMES, build_sim_features, ready_counts,
                                    sec_since_last)
from src.sim_track.targets import known_target, make_sim_folds


@pytest.fixture(scope="module")
def sim():
    from src.simulator.config import SimConfig
    from src.simulator.world import simulate
    res = simulate(SimConfig.small())
    tx = res.transactions.sort_values(["event_time", "transaction_id"]).reset_index(drop=True)
    return tx, res.events


@pytest.fixture(scope="module")
def built(sim):
    tx, ev = sim
    return build_sim_features(tx, ev)


def test_features_build_and_exclude_truth_and_outcomes(sim, built):
    tx, _ = sim
    out, feats = built
    assert len(out) == len(tx)
    assert not set(feats) & set(GROUND_TRUTH + OUTCOMES + ["transaction_id", "event_time"])
    assert not any(c.startswith(BURST_PREFIX) for c in feats)
    assert any(c.startswith(BURST_PREFIX) for c in out.columns)
    assert all(out[c].dtype.kind == "f" for c in feats)
    for c in ["merchant_id_fraud_rate", "merchant_ramp_1d_vs_30d", "merchant_refund_ratio_7d",
              "acct_sec_since_cred_change", "acct_failed_logins_1d", "acct_device_unseen_in_events",
              "device_id_distinct_card_id_1d", "kmh_from_prev_geo", "cluster_n_customers", "customer_id_new_device_id"]:
        assert c in feats
    assert out.columns.is_unique


def test_features_never_see_the_future(sim, built):
    """Change EVERYTHING after a cut-off (amounts, approvals, labels, label dates, truth, events):
    features of rows before the cut-off must not change."""
    tx, ev = sim
    out1, feats = built
    cut = tx["event_time"].quantile(0.6)
    w2, e2 = tx.copy(), ev.copy()
    late = w2["event_time"] > cut
    w2.loc[late, "amount_usd"] *= 9
    w2.loc[late, "approved"] = 1 - w2.loc[late, "approved"]
    w2.loc[late, "reported_label"] = 1 - w2.loc[late, "reported_label"].fillna(0)
    w2.loc[late, "is_fraud_true"] = 1 - w2.loc[late, "is_fraud_true"]
    w2.loc[late, "label_date"] = w2.loc[late, "event_time"] + pd.Timedelta(seconds=1)
    el = e2["event_time"] > cut
    e2.loc[el, "success"] = ~e2.loc[el, "success"].astype(bool)
    out2, _ = build_sim_features(w2, e2)
    early = (out1["event_time"] <= cut).to_numpy()
    cols = feats + [c for c in out1.columns if c.startswith(BURST_PREFIX)]
    pd.testing.assert_frame_equal(out1.loc[early, cols].reset_index(drop=True),
                                  out2.loc[early, cols].reset_index(drop=True), check_dtype=False)


def test_features_ignore_true_label_entirely(sim, built):
    """is_fraud_true may be anything: features are a function of reported_label / label_date only."""
    tx, ev = sim
    out1, feats = built
    w2 = tx.copy()
    w2["is_fraud_true"] = 1 - w2["is_fraud_true"]
    w2["is_fraud"] = w2["is_fraud_true"]
    w2["fraud_scenario"] = "legit"
    out2, _ = build_sim_features(w2, ev)
    pd.testing.assert_frame_equal(out1[feats], out2[feats], check_dtype=False)


def test_ready_counts_and_event_lookup():
    codes = np.array([0, 0, 0, 0])
    t = np.array([0, 10, 20, 30])
    ready = np.array([25.0, np.nan, 5.0, np.nan])
    mask = np.array([True, False, True, False])
    # row0 ready@25, row2 ready@5: before t=0:0, t=10:1 (row2), t=20:1, t=30:2
    np.testing.assert_array_equal(ready_counts(codes, t, mask, ready), [0, 1, 1, 2])
    s = sec_since_last(np.array([1, 1, 2]), np.array([100, 200, 50]), np.array([1, 1, 2, 3, -1]),
                       np.array([150, 200, 60, 10, 10]))
    np.testing.assert_array_equal(s[:3], [50, 100, 10])          # event strictly before the row
    assert np.isnan(s[3]) and np.isnan(s[4])


def test_targets_only_use_labels_known_at_cutoff(sim):
    tx, _ = sim
    cutoff = tx["event_time"].quantile(0.8)
    y1, u1 = known_target(tx["reported_label"], tx["label_date"], tx["event_time"], cutoff, gap_days=7)
    # destroy everything that is only knowable after the cutoff -> target must not change
    r2, d2 = tx["reported_label"].copy(), tx["label_date"].copy()
    later = (d2 > cutoff) & d2.notna()
    r2[later] = 1 - r2[later].fillna(0)
    y2, u2 = known_target(r2, d2, tx["event_time"], cutoff, gap_days=7)
    pos_known = ((tx["reported_label"] == 1) & (tx["label_date"] <= cutoff)).to_numpy()
    assert (y1[pos_known] == 1).all()
    assert y1[(tx["label_date"] > cutoff).to_numpy()].sum() == 0      # pending fraud is NOT a positive yet
    np.testing.assert_array_equal(u1, u2)
    # the truth column is not even an argument; a label flip after the cutoff cannot create positives
    assert y2[later.to_numpy()].sum() == 0
    # unobserved rows (no label_date) are excluded, gap rows are excluded
    assert not u1[tx["label_date"].isna().to_numpy()].any()
    assert not u1[(tx["event_time"] >= cutoff - pd.Timedelta(days=7)).to_numpy()].any()
    assert u1.sum() > 0


def test_unobserved_rows_never_in_training_folds(sim):
    tx, _ = sim
    end = tx["event_time"].max()
    folds = make_sim_folds(tx, end, n_folds=2, val_days=7, gap_days=3)
    assert folds
    unobs = tx["label_date"].isna().to_numpy()
    for f in folds:
        assert not unobs[f.train].any()
        assert (tx["event_time"].iloc[f.train] < f.cutoff - pd.Timedelta(days=3)).all()
        assert (tx["event_time"].iloc[f.score] >= f.cutoff).all()
        assert set(f.es).isdisjoint(set(f.score))


def test_sim_report_both_truths_and_slices(sim, built):
    out, _ = built
    rng = np.random.default_rng(0)
    p = out["is_fraud_true"].to_numpy() * 0.5 + rng.random(len(out)) * 0.6        # a decent fake scorer
    rep = sim_report(out, p, cp_k=20)
    for k in ("true", "reported"):
        assert rep[k]["auc_pr"] > 0
        assert "false_decline_rate_top1pct" in rep[k]
    assert "by_scenario" in rep and "novel" in rep and "slices" in rep
    assert rep["true"]["n"] == len(out)
    assert rep["reported"]["n"] < rep["true"]["n"]               # rows without an outcome are not measurable
    x = pct_vs_reference(np.arange(10.0), np.array([-1.0, 4.5, 99.0]))
    assert x[0] == 0 and x[2] == 1

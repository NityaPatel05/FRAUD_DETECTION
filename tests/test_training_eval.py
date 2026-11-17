"""Blend ranking/calibration, operational metrics (CP@k, VDR), slices, ablation verdict."""
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.training.blending import CalibratedRankBlender
from src.training.evaluation import (
    card_precision_at_k, copied_positive_flags, mean_se, score_report, slice_masks, sliced_report,
    threshold_at_fpr, value_detection_rate,
)

ROOT = Path(__file__).resolve().parent.parent


# ── blend ─────────────────────────────────────────────────────────────────────

def _scores(n=30_000, seed=0):
    r = np.random.default_rng(seed)
    y = (r.random(n) < 0.04).astype(int)
    a = 1 / (1 + np.exp(-(3 * y - 3 + r.normal(0, 1.2, n))))
    b = 1 / (1 + np.exp(-(2.5 * y - 3 + r.normal(0, 1.2, n))))
    return y, a, b


def test_blend_ranking_is_tie_free_and_calibration_is_monotone_in_the_score():
    y, a, b = _scores()
    bl = CalibratedRankBlender({"a": 0.5, "b": 0.5}).fit({"a": a[:20000], "b": b[:20000]}, y[:20000])
    pr = {"a": a[20000:], "b": b[20000:]}
    s, p = bl.score(pr), bl.predict_proba(pr)
    assert len(np.unique(s)) == len(s)                                   # continuous ranking score
    assert len(np.unique(p)) == len(p)                                   # no isotonic plateau ties (was 26% at p=0)
    order = np.argsort(s)
    assert np.all(np.diff(p[order]) > 0)                                 # strictly monotone: ranking preserved
    assert np.mean(p == 0) == 0
    from sklearn.metrics import average_precision_score as ap
    assert ap(y[20000:], p) == pytest.approx(ap(y[20000:], s), abs=1e-9)  # serving probability ranks like the score
    assert abs(p.mean() - y[20000:].mean()) < 0.01                       # still calibrated on average


def test_percentile_saturation_guard_for_more_confident_refit_models():
    y, a, b = _scores()
    bl = CalibratedRankBlender({"a": 0.5, "b": 0.5}).fit({"a": a, "b": b}, y)
    top = float(max(a.max(), b.max()))
    hot = {"a": np.array([top + 0.001, top + 0.002, top + 0.01]), "b": np.array([top + 0.001, top + 0.002, top + 0.01])}
    s = bl.score(hot)
    assert np.all(np.diff(s) > 0) and s.max() < 1.0                      # beyond the OOF range: still strictly increasing
    p = bl.predict_proba(hot)
    assert np.all(np.diff(p) > 0)
    low = bl.score({"a": np.array([-5.0, -4.0]), "b": np.array([-5.0, -4.0])})
    assert low[0] < low[1] and low.min() > 0


def test_blender_pickle_roundtrip_and_single_transaction(tmp_path):
    y, a, b = _scores(8000)
    bl = CalibratedRankBlender({"a": 0.7, "b": 0.3, "zero": 0.0}).fit({"a": a, "b": b}, y)
    assert "zero" not in bl.weights
    bl.save(tmp_path / "b.pkl")
    bl2 = CalibratedRankBlender.load(tmp_path / "b.pkl")
    np.testing.assert_allclose(bl2.predict_proba({"a": a, "b": b}), bl.predict_proba({"a": a, "b": b}))
    one = bl.predict_proba({"a": np.array([0.9]), "b": np.array([0.9])})
    assert one.shape == (1,)
    # old pickles (no extra attributes) keep working: only refs/weights/calibrator are needed
    assert set(bl.__dict__) == {"weights", "max_ref", "seed", "refs", "calibrator"}
    assert pickle.loads(pickle.dumps(bl)).predict_proba({"a": a[:5], "b": b[:5]}).shape == (5,)


# ── operational metrics ───────────────────────────────────────────────────────

def test_card_precision_at_k_per_day_removes_detected_clients():
    client = np.array([1, 2, 3, 1, 2, 3, 4])
    day = np.array([1, 1, 1, 2, 2, 2, 2])
    y = np.array([1, 0, 0, 1, 1, 0, 0])
    p = np.array([.9, .8, .1, .95, .7, .6, .2])
    cp, ndays = card_precision_at_k(y, p, client, day, k=2)
    # day 1: top2 = clients 1,2 -> 1 fraud -> 0.5; client 1 is blocked afterwards.
    # day 2: clients 2,3,4 left; top2 = 2 (fraud) and 3 -> 0.5.  Without removal day 2 would be 1.0.
    assert ndays == 2 and cp == pytest.approx(0.5)
    c_unknown = np.array([-1, 2, 3, 1, 2, 3, 4])                         # rows without client are ignored
    cp2, _ = card_precision_at_k(y, p, c_unknown, day, k=2)
    assert np.isfinite(cp2)


def test_value_detection_rate_at_fixed_fpr():
    legit = np.linspace(0, 0.99, 100)                                    # 100 legit rows
    y = np.concatenate([np.zeros(100), [1, 1, 1]]).astype(int)
    p = np.concatenate([legit, [2.0, 0.995, 0.5]])
    amt = np.concatenate([np.full(100, 10.0), [100.0, 300.0, 600.0]])
    # 1% FPR of 100 legit = 1 allowed false positive: threshold 0.99 catches the 100 and 300 frauds
    assert threshold_at_fpr(y, p, 0.01) == pytest.approx(0.99)
    assert value_detection_rate(y, p, amt, 0.01) == pytest.approx((100 + 300) / 1000)
    assert value_detection_rate(y, p, amt, 0.0) == pytest.approx(0.4)    # both top frauds outscore every legit row
    assert value_detection_rate(y, p, amt, 0.5) == pytest.approx(1.0)    # 50% FPR also catches the 600 fraud
    rep = score_report(y, p, amount=amt)
    assert rep["vdr_at_fpr_1pct"] == pytest.approx(0.4) and "vdr_at_fpr_0.5pct" in rep
    assert rep["false_decline_rate_top1pct"] >= 0 and rep["ap_over_base_rate"] > 1


def test_score_report_cp_and_false_decline():
    r = np.random.default_rng(0)
    n = 3000
    y = (r.random(n) < 0.05).astype(int)
    p = y * 0.5 + r.random(n)
    client = r.integers(0, 400, n)
    day = r.integers(0, 10, n)
    rep = score_report(y, p, client=client, amount=np.full(n, 20.0), day=day, k=20)
    assert 0 <= rep["cp_at_20_daily"] <= 1 and rep["cp_days"] > 0
    flagged_legit = rep["false_decline_rate_top1pct"] * (y == 0).sum()
    assert flagged_legit <= round(n * 0.01) + 1e-6


def test_copied_positive_flags_and_slices():
    # client 1: frauds at t=1 (first), t=3 (copy), legit t=2 ; client 2: one fraud (not a copy) ; client -1 never flagged
    client = np.array([1, 1, 1, 2, -1, -1])
    t = np.array([1, 2, 3, 4, 5, 6])
    ids = np.arange(6)
    y = np.array([1, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(copied_positive_flags(client, t, ids, y), [False, False, True, False, False, False])
    m = slice_masks(client_score=np.array([1, 2, 7, -1]), client_fit=np.array([1, 2, 3, -1]),
                    copied_score=np.array([False, True, False, False]))
    np.testing.assert_array_equal(m["client_seen"], [True, True, False, False])
    np.testing.assert_array_equal(m["cold_start"], [False, False, True, False])
    np.testing.assert_array_equal(m["uid_unbuildable"], [False, False, False, True])
    np.testing.assert_array_equal(m["copied_removed"], [True, False, True, True])


def test_sliced_report_skips_tiny_slices():
    r = np.random.default_rng(1)
    y = (r.random(2000) < 0.1).astype(int)
    p = y + r.random(2000)
    masks = {"all": np.ones(2000, bool), "tiny": np.arange(2000) < 10}
    out = sliced_report(y, p, masks)
    assert out["all"]["auc_pr"] > 0.5 and "skipped" in out["tiny"]


def test_mean_se():
    m, se = mean_se([1.0, 2.0, 3.0])
    assert m == 2.0 and se == pytest.approx(1 / np.sqrt(3))
    assert mean_se([5.0]) == (5.0, 0.0)


# ── ablation verdict ──────────────────────────────────────────────────────────

def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ablation_verdict_needs_gain_over_two_se_and_threshold():
    ab = _load_script("run_ablation")
    # clear gain: mean .02, tiny noise
    assert ab.ablation_verdict([0.019, 0.021, 0.02, 0.02, 0.02])[2] == "KEEP"
    # above 0.005 on average but noise (SE) is huge -> not provable
    mean, se, v = ab.ablation_verdict([0.04, -0.02, 0.03, -0.02, 0.0, 0.0, 0.04, 0.0])
    assert mean >= 0.005 and mean <= 2 * se and v.startswith("inconclusive")
    # very consistent but below threshold -> drop (the old rule's noise floor)
    assert ab.ablation_verdict([0.003, 0.0031, 0.0029, 0.003, 0.003])[2].startswith("drop")
    # removing the family helps, consistently
    assert ab.ablation_verdict([-0.01, -0.011, -0.009, -0.01, -0.01])[2].startswith("HURTS")
    groups = ab.parse_groups(["overlap", "a,b"])
    assert groups["behaviour+behaviour_v3+hmm_sequence+graph+network_v4"] == [
        "behaviour", "behaviour_v3", "hmm_sequence", "graph", "network_v4"] and groups["a+b"] == ["a", "b"]


# ── label encoder fitted on the train window only ─────────────────────────────

def test_label_encoder_is_fitted_on_train_rows_only():
    import pandas as pd
    from src.feature_engineering.base_features import label_encode_strings
    df = pd.DataFrame({"c": ["b", "a", "z", None, "a", "q"], "split": ["train"] * 3 + ["val"] * 3})
    fit = (df["split"] == "train").to_numpy()
    out, maps = label_encode_strings(df.copy(), exclude=["split"], fit_mask=fit)
    assert maps["c"] == ["a", "b", "z"]                      # only train categories, sorted
    enc = out["c"].to_numpy()
    assert enc[0] == 1 and enc[1] == 0 and enc[2] == 2       # b, a, z
    assert np.isnan(enc[3])                                   # missing stays missing
    assert enc[4] == 0                                        # 'a' known from train
    assert np.isnan(enc[5])                                   # 'q' first seen after train -> missing, like online

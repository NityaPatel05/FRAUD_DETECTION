"""Label policy (src/labels/policy.py): determinism, no fraud known early, maturity, feature
counters vs a brute-force reference, training targets as known at a cutoff, rolling folds."""
import numpy as np
import pytest

from src.feature_engineering.entity_ops import delayed_label_stats, policy_label_stats
from src.feature_engineering.streaming_ops import identity_clusters, identity_clusters_policy
from src.labels.policy import (
    DAY, LabelPolicy, fraud_delay_sec, fraud_known_at, fraud_known_by, is_mature, known_label,
    label_ready_times, maturity_at, training_target,
)
from src.training.evaluation import make_folds

POL = LabelPolicy()


def test_delay_is_deterministic_and_matches_scalar_and_array():
    ids = np.arange(1_000_000, 1_000_500)
    a, b = fraud_delay_sec(ids, POL), fraud_delay_sec(ids.copy(), POL)
    np.testing.assert_array_equal(a, b)
    assert fraud_delay_sec(int(ids[7]), POL) == a[7]                  # online scalar == offline array
    assert fraud_known_at(int(ids[7]), 5_000, POL) == 5_000 + a[7]
    assert not np.array_equal(a, fraud_delay_sec(ids, POL.replace(seed=1)))   # seed matters
    assert fraud_delay_sec(5, POL) == fraud_delay_sec(np.array([5]), POL)[0]


def test_delay_distribution_median_and_cap():
    d = fraud_delay_sec(np.arange(0, 200_000), POL) / DAY
    assert 33 < np.median(d) < 37
    assert d.min() >= POL.delay_min_days and d.max() <= POL.delay_cap_days
    assert (d >= POL.delay_cap_days).mean() > 0           # the cap is hit by the heavy tail
    fixed = fraud_delay_sec(np.arange(50), LabelPolicy.fixed(10)) / DAY
    assert np.all(fixed == 10)


def test_no_fraud_known_before_t_plus_d_and_maturity():
    ids = np.arange(1, 2001)
    t = np.full(len(ids), 1_000_000)
    d = fraud_delay_sec(ids, POL)
    y_fraud = np.ones(len(ids))
    just_before = t + d - 1
    assert np.all(np.isnan(known_label(y_fraud, ids, t, just_before, POL)[just_before < maturity_at(t, POL)]))
    assert not fraud_known_by(ids, t, just_before, POL).any()
    assert np.all(known_label(y_fraud, ids, t, t + d, POL) == 1.0)
    # legit rows: pending until maturity, then 0
    y0 = np.zeros(len(ids))
    assert np.all(np.isnan(known_label(y0, ids, t, maturity_at(t, POL) - 1, POL)))
    assert np.all(known_label(y0, ids, t, maturity_at(t, POL), POL) == 0.0)
    assert is_mature(0, 90 * DAY, POL) and not is_mature(0, 90 * DAY - 1, POL)


def test_late_fraud_is_seen_as_legit_first():
    ids = np.arange(1, 5001)
    d = fraud_delay_sec(ids, POL)
    late = ids[d > POL.maturity_days * DAY][:5]
    assert len(late) > 0
    t = np.zeros(len(late), dtype=np.int64)
    mid = maturity_at(t, POL) + 1                              # mature, chargeback not yet there
    assert np.all(known_label(np.ones(len(late)), late, t, mid, POL) == 0.0)
    after = t + fraud_delay_sec(late, POL)
    assert np.all(known_label(np.ones(len(late)), late, t, after, POL) == 1.0)


def _random_entities(n=400, n_ent=15, seed=0):
    r = np.random.default_rng(seed)
    t = np.sort(r.integers(0, 200 * DAY, n))
    codes = r.integers(0, n_ent, n)
    codes[r.random(n) < 0.05] = -1
    y = (r.random(n) < 0.15).astype(float)
    y[r.random(n) < 0.1] = np.nan                               # unlabelled rows (e.g. Kaggle test)
    ids = np.arange(n) + 10_000
    return codes.astype(np.int64), t.astype(np.int64), y, ids


def test_policy_label_stats_matches_bruteforce_and_never_sees_self_or_future():
    codes, t, y, ids = _random_entities()
    ev = label_ready_times(y, ids, t, POL)
    st = policy_label_stats(codes, t, ev)
    for i in range(0, len(t), 7):
        if codes[i] < 0:
            assert np.isnan(st["n"][i]); continue
        same = (codes == codes) & (codes == codes[i]) & (np.arange(len(t)) != i) & ~np.isnan(y)
        n = (same & is_mature(t, t[i], POL)).sum()
        f = (same & (y == 1) & fraud_known_by(ids, t, t[i], POL)).sum()
        fm = (same & (y == 1) & fraud_known_by(ids, t, t[i], POL) & is_mature(t, t[i], POL)).sum()
        assert (st["n"][i], st["fraud"][i], st["fraud_mature"][i]) == (n, f, fm)
    # changing the row's own label / a future label changes nothing
    y2 = y.copy(); y2[100] = 1 - np.nan_to_num(y2[100]); y2[300:] = 1.0
    st2 = policy_label_stats(codes, t, label_ready_times(y2, ids, t, POL))
    keep = np.arange(len(t)) <= 100
    np.testing.assert_array_equal(st["n"][keep], st2["n"][keep])
    np.testing.assert_array_equal(st["fraud"][keep], st2["fraud"][keep])


def test_fixed_policy_reproduces_legacy_delayed_label_stats_and_clusters():
    codes, t, y, ids = _random_entities(seed=3)
    pol = LabelPolicy.fixed(30)
    ev = label_ready_times(y, ids, t, pol)
    new = policy_label_stats(codes, t, ev)
    old = delayed_label_stats(codes, t, y, 30 * DAY)
    np.testing.assert_array_equal(new["n"], old["n"])
    np.testing.assert_array_equal(new["fraud"], old["fraud"])
    client = np.where(codes >= 0, codes, -1)
    link = np.random.default_rng(1).integers(-1, 8, len(t)).astype(np.int64)
    newp = np.ones(len(t))
    a = identity_clusters(t, client, link, newp, y, 30 * DAY, hub_cap=5)
    b = identity_clusters_policy(t, client, link, newp, ev, hub_cap=5)
    for k in ("size", "fraud", "labelled", "merge"):
        np.testing.assert_array_equal(a[k], b[k])


def test_training_target_masks_unknown_fraud_and_gap():
    ids = np.arange(1, 3001)
    t = np.repeat(np.arange(0, 100) * DAY, 30).astype(np.int64)
    y = np.ones(len(ids))                                      # everything is fraud: only the known ones count
    cutoff = 100 * DAY
    target, usable = training_target(y, ids, t, cutoff, POL, gap_days=7)
    assert not usable[t >= cutoff - 7 * DAY].any()             # rows in the gap are excluded
    assert usable[t < cutoff - 7 * DAY].all()
    known = fraud_known_by(ids, t, cutoff, POL)
    np.testing.assert_array_equal(target.astype(bool), known)  # 1 iff the fraud label is known by the cutoff
    assert 0 < target[usable].mean() < 1                       # some frauds still in their chargeback delay
    # an earlier cutoff knows strictly fewer frauds
    t2, u2 = training_target(y, ids, t, cutoff - 30 * DAY, POL, gap_days=7)
    assert t2[u2 & usable].sum() <= target[u2 & usable].sum()


def test_make_folds_train_gap_val_and_disjoint_es_score():
    n_days = 130
    t = (np.repeat(np.arange(1, n_days + 1), 20) * DAY + 100).astype(np.int64)
    ids = np.arange(len(t)) + 1
    y = (np.random.default_rng(0).random(len(t)) < 0.05).astype(int)
    folds = make_folds(t, y, ids, n_folds=5, val_days=10, policy=POL, gap_days=7, es_frac=0.5)
    assert len(folds) == 5
    day = t // DAY
    for f in folds:
        assert day[f.train].max() < f.cutoff // DAY - 7 + 1             # train | gap
        assert f.cutoff // DAY - day[f.train].max() > 7 - 1
        assert set(f.es).isdisjoint(f.score)                            # early stopping never on scored rows
        assert day[f.es].max() < day[f.score].min()                     # ES part first in time
        assert day[f.score].min() >= f.val_start_day and day[f.score].max() < f.val_end_day
        np.testing.assert_array_equal(f.y_train, training_target(y, ids, t, f.cutoff, POL, 7)[0][f.train])
        assert f.y_train.sum() <= y[f.train].sum()                      # target never exceeds the truth
    assert folds[-1].val_end_day == n_days + 1
    assert folds[1].val_start_day == folds[0].val_end_day               # rolling windows tile the tail


def test_policy_from_cfg_roundtrip():
    p = LabelPolicy.from_cfg({"delay_median_days": 60, "maturity_days": 120, "unknown_key": 1})
    assert p.delay_median_days == 60 and p.maturity_days == 120
    assert LabelPolicy.from_cfg(p.to_dict()) == p
    with pytest.raises(ValueError):
        LabelPolicy(delay_min_days=0)

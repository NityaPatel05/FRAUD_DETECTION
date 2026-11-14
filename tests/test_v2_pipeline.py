"""
Tests for the v2 feature/training code. Synthetic data only — no real data, no model training.

Key property tested: causal features of a row never change when FUTURE rows change
(no time leakage), and label-history features never see labels younger than the delay.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering.entity_ops import (
    make_codes, entity_mean_std, entity_nunique, window_count_sum,
    delayed_label_stats, first_time_pair, prev_row_values,
)
from src.training.evaluation import time_folds, client_smoothing, rank_blend, score_report

RNG = np.random.default_rng(0)


def _entities(n=400, n_ent=25):
    codes = RNG.integers(0, n_ent, n).astype(np.int64)
    codes[RNG.random(n) < 0.05] = -1                       # some unknown keys
    t = np.sort(RNG.integers(0, 60 * 86400, n)).astype(np.int64)
    t[10] = t[11]                                          # a timestamp tie
    vals = RNG.normal(100, 30, n)
    vals[RNG.random(n) < 0.1] = np.nan
    return codes, t, vals


# ─────────────────────────────────────────────────────────────────────────────
# entity_ops vs brute force
# ─────────────────────────────────────────────────────────────────────────────

def test_causal_mean_std_matches_bruteforce():
    codes, t, v = _entities()
    out = entity_mean_std(codes, t, v, "causal")
    for i in range(len(codes)):
        if codes[i] < 0:
            assert np.isnan(out["mean"][i]); continue
        prior = [v[j] for j in range(i + 1) if codes[j] == codes[i] and not np.isnan(v[j])]
        if prior:
            assert out["mean"][i] == pytest.approx(np.mean(prior))
        if len(prior) > 1:
            assert out["std"][i] == pytest.approx(np.std(prior, ddof=1), rel=1e-6)


def test_exclude_current_uses_strictly_prior_rows():
    codes, t, v = _entities()
    out = entity_mean_std(codes, t, v, "causal", include_current=False)
    for i in range(len(codes)):
        if codes[i] < 0:
            continue
        prior = [v[j] for j in range(i) if codes[j] == codes[i] and not np.isnan(v[j])]
        if prior:
            assert out["mean"][i] == pytest.approx(np.mean(prior))
        else:
            assert np.isnan(out["mean"][i])


def test_window_count_sum_matches_bruteforce():
    codes, t, v = _entities()
    w = 3 * 86400
    out = window_count_sum(codes, t, w, values=v)
    for i in range(len(codes)):
        if codes[i] < 0:
            assert np.isnan(out["count"][i]); continue
        js = [j for j in range(i + 1) if codes[j] == codes[i] and t[j] > t[i] - w]
        assert out["count"][i] == len(js)
        assert out["sum"][i] == pytest.approx(np.nansum([v[j] for j in js]))


def test_causal_nunique_and_first_time():
    codes, _, _ = _entities()
    t = np.arange(len(codes)) * 100
    vc = RNG.integers(-1, 5, len(codes))
    nun = entity_nunique(codes, t, vc, "causal")
    ft = first_time_pair(codes, t, vc)
    for i in range(len(codes)):
        if codes[i] < 0:
            continue
        seen = {vc[j] for j in range(i + 1) if codes[j] == codes[i] and vc[j] >= 0}
        assert nun[i] == len(seen)
        has_prior = any(codes[j] == codes[i] for j in range(i))
        if vc[i] >= 0 and has_prior:
            before = {vc[j] for j in range(i) if codes[j] == codes[i]}
            assert ft[i] == float(vc[i] not in before)


def test_prev_row_values():
    codes = np.array([0, 1, 0, 0, 1])
    t = np.array([1, 2, 3, 4, 5])
    v = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    np.testing.assert_array_equal(prev_row_values(codes, t, v), [np.nan, np.nan, 10.0, 30.0, 20.0])


def test_delayed_labels_respect_delay_and_never_see_own_label():
    codes = np.zeros(5, dtype=np.int64)
    day = 86400
    t = np.array([0, 1, 5, 40, 41]) * day
    y = np.array([1.0, 0.0, 1.0, np.nan, 1.0])          # row 3 unlabelled
    st = delayed_label_stats(codes, t, y, delay_sec=30 * day)
    # rows 0-2: nothing is 30 days old yet
    np.testing.assert_array_equal(st["n"][:3], [0, 0, 0])
    # row 3 (day 40): rows at day 0,1,5 are ≥30 days old → 3 labelled, 2 fraud
    assert st["n"][3] == 3 and st["fraud"][3] == 2
    assert st["n"][4] == 3 and st["fraud"][4] == 2    # day 41: day 40 row not old enough
    with pytest.raises(ValueError):
        delayed_label_stats(codes, t, y, delay_sec=0)


def test_transductive_uses_all_rows():
    codes = np.array([0, 0, 0])
    t = np.array([1, 2, 3])
    v = np.array([1.0, 2.0, 6.0])
    assert entity_mean_std(codes, t, v, "transductive")["mean"][0] == pytest.approx(3.0)
    assert entity_mean_std(codes, t, v, "causal")["mean"][0] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Full v2 pipeline on a synthetic IEEE-like frame
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_ieee(n, seed, start_id=0, labelled=True):
    r = np.random.default_rng(seed)
    dt = np.sort(r.integers(86400, 86400 * 90, n))
    df = pd.DataFrame({
        "TransactionID": np.arange(start_id, start_id + n),
        "TransactionDT": dt,
        "TransactionAmt": np.round(r.lognormal(4, 1, n), 3),
        "ProductCD": r.choice(list("WCHRS"), n),
        "card1": r.integers(1000, 1030, n),
        "card2": r.choice([100.0, 200.0, np.nan], n),
        "card3": 150.0, "card4": r.choice(["visa", "mastercard"], n),
        "card5": r.choice([226.0, 102.0], n), "card6": r.choice(["debit", "credit"], n),
        "addr1": r.choice([100.0, 200.0, 300.0, np.nan], n, p=[.4, .3, .25, .05]),
        "addr2": r.choice([87.0, 87.0, 60.0, np.nan], n), "dist1": r.choice([1.0, 5.0, np.nan], n),
        "P_emaildomain": r.choice(["gmail.com", "yahoo.com", "anonymous.com", None], n),
        "R_emaildomain": r.choice(["gmail.com", None], n),
        "DeviceType": r.choice(["desktop", "mobile", None], n),
        "DeviceInfo": r.choice(["Windows", "iOS Device", "SM-G9", None], n),
        "id_19": r.choice([100.0, 200.0, np.nan], n), "id_20": r.choice([300.0, np.nan], n),
        "id_30": r.choice(["Windows 10", "iOS 11.1", None], n),
        "id_31": r.choice(["chrome 63.0", "safari", None], n), "id_33": r.choice(["1920x1080", None], n),
        "id_12": r.choice(["Found", "NotFound", None], n),
    })
    day = df["TransactionDT"] // 86400
    df["D1"] = (day - r.integers(0, 40, n)).clip(lower=0).astype(float)
    for k in range(2, 16):
        df[f"D{k}"] = r.choice([np.nan, 1.0, 10.0, 50.0], n)
    for k in range(1, 15):
        df[f"C{k}"] = r.integers(0, 5, n).astype(float)
    for k in range(1, 10):
        df[f"M{k}"] = r.choice(["T", "F", None], n)
    df["M4"] = r.choice(["M0", "M1", "M2", None], n)
    for k in range(1, 13):
        base = r.normal(size=n)
        df[f"V{k}"] = base if k % 3 else base * 2 + 0.01 * r.normal(size=n)  # correlated pairs
    if labelled:
        df["isFraud"] = (r.random(n) < 0.08).astype(int)
    return df


@pytest.fixture(scope="module")
def v2_output():
    from src.feature_engineering.pipeline_v2 import run_feature_pipeline_v2
    full = _synthetic_ieee(3000, 1)
    tr = full[full["TransactionDT"] < 86400 * 60]
    va = full[full["TransactionDT"] >= 86400 * 60]
    te = _synthetic_ieee(800, 2, start_id=10_000, labelled=False)
    te["TransactionDT"] += 86400 * 90
    te.columns = [c.replace("id_", "id-") if c.startswith("id_") else c for c in te.columns]  # Kaggle quirk
    data, meta = run_feature_pipeline_v2(tr, va, te, {"label_delay_days": 10})
    return data, meta, (tr, va, te)


def test_v2_feature_frame_is_clean(v2_output):
    data, meta, _ = v2_output
    feats = meta["feature_cols"]
    for bad in ["TransactionDT", "TransactionID", "day", "isFraud", "uid1", "uid2", "uid1_code", "split"]:
        assert bad not in feats
    for part in data.values():
        assert all(part[c].dtype.kind in "fiub" for c in feats), "string column leaked into features"
    # previously-dropped categorical columns now reach the model
    for c in ["ProductCD", "card4", "card6", "P_emaildomain", "M4", "id_12"]:
        assert c in feats
    assert data["test"]["id_19"].notna().any(), "test identity columns lost (id- vs id_)"
    assert {"uid1_lbl_rate", "uid_TransactionAmt_mean", "amt_z_vs_own_past", "uid_n_1d",
            "device_fp_n_uid1"} <= set(feats)
    assert data["test"]["isFraud"].isna().all()


def test_v2_causal_features_do_not_see_future(v2_output):
    """Changing future rows (values and labels) must not change earlier rows' features."""
    from src.feature_engineering.pipeline_v2 import run_feature_pipeline_v2
    data, meta, (tr, va, te) = v2_output
    va2 = va.copy()
    cut = va2["TransactionDT"].min() + 86400 * 10
    late = va2["TransactionDT"] > cut
    va2.loc[late, "TransactionAmt"] *= 7
    va2.loc[late, "isFraud"] = 1 - va2.loc[late, "isFraud"]
    va2.loc[late, "C1"] = 99
    data2, _ = run_feature_pipeline_v2(tr, va2, te, {"label_delay_days": 10})
    feats = [c for c in meta["feature_cols"] if not c.endswith("_count30d")]  # window counts: also causal
    a = pd.concat([data["train"], data["val"]]).set_index("TransactionID")
    b = pd.concat([data2["train"], data2["val"]]).set_index("TransactionID")
    early_ids = a.index[a["TransactionDT"] <= cut]
    pd.testing.assert_frame_equal(a.loc[early_ids, feats], b.loc[early_ids, feats], check_dtype=False)


def test_leakage_test_is_sensitive(v2_output):
    """Control: the same future-change check MUST detect the (intended) future
    dependence of transductive mode — proves the causal test above has teeth."""
    from src.feature_engineering.pipeline_v2 import run_feature_pipeline_v2
    _, _, (tr, va, te) = v2_output
    cfg = {"agg_mode": "transductive", "label_delay_days": 10}
    d1, meta = run_feature_pipeline_v2(tr, va, te, cfg)
    va2 = va.copy()
    cut = va2["TransactionDT"].min() + 86400 * 10
    va2.loc[va2["TransactionDT"] > cut, "TransactionAmt"] *= 7
    d2, _ = run_feature_pipeline_v2(tr, va2, te, cfg)
    a = d1["train"].set_index("TransactionID")["uid_TransactionAmt_mean"]
    b = d2["train"].set_index("TransactionID")["uid_TransactionAmt_mean"]
    assert not np.allclose(a.fillna(-1), b.fillna(-1))


def test_v2_transductive_mode_runs(v2_output):
    from src.feature_engineering.pipeline_v2 import run_feature_pipeline_v2
    _, _, (tr, va, te) = v2_output
    data, meta = run_feature_pipeline_v2(tr, va, te, {"agg_mode": "transductive", "label_delay_days": 10})
    assert "card1_count" in meta["feature_cols"]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_time_folds_are_forward_with_gap():
    day = np.repeat(np.arange(1, 131), 10)
    folds = time_folds(day, n_folds=3, val_days=15, gap_days=7)
    assert len(folds) == 3
    for tr, va in folds:
        assert day[tr].max() + 7 < day[va].min()
        assert day[va].max() - day[va].min() == 14
    assert day[folds[-1][1]].max() == 130


def test_client_smoothing_causal_has_no_future():
    p = np.array([0.1, 0.9, 0.5, 0.2])
    c = np.array([0, 0, 1, 0])
    t = np.array([1, 2, 3, 4])
    s = client_smoothing(p, c, t, "causal", alpha=1.0)
    np.testing.assert_allclose(s, [0.1, 0.5, 0.5, 0.4])
    p2 = p.copy(); p2[3] = 1.0
    assert client_smoothing(p2, c, t, "causal", alpha=1.0)[1] == pytest.approx(0.5)


def test_rank_blend_and_report():
    y = np.array([0, 0, 1, 1, 0, 1])
    a = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    b = a * 100
    np.testing.assert_allclose(rank_blend({"a": a, "b": b}, {"a": 0.5, "b": 0.5}),
                               rank_blend({"a": a}, {"a": 1.0}))
    rep = score_report(y, a, client=np.array([0, 1, 2, 3, 4, 5]))
    assert rep["auc_pr"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Bug fixes in legacy code
# ─────────────────────────────────────────────────────────────────────────────

def test_uid_key_batch_equals_inference():
    from src.feature_engineering.uid_features import build_uid_features, build_uid_features_inference
    df = pd.DataFrame({"card1": [13926], "addr1": [315.0], "D1": [14.0], "TransactionDT": [86400],
                       "P_emaildomain": ["gmail.com"], "DeviceInfo": ["Windows"]})
    df["day"] = df["TransactionDT"] // 86400
    build_uid_features(df)
    inf = build_uid_features_inference(df.iloc[0].to_dict())
    assert df.loc[0, "uid1"] == inf["uid1"] == "13926_315_13"


def test_m4_encoding():
    from src.preprocessing.missing_values import encode_m_columns
    df = pd.DataFrame({"M4": ["M0", "M2", None], "M1": ["T", "F", None]})
    encode_m_columns(df, nan_value=-1)
    assert df["M4"].tolist() == [0, 2, -1]
    assert df["M1"].tolist() == [1, 0, -1]


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2-4 additions: streaming primitives, derived signals, raw schema
# ═════════════════════════════════════════════════════════════════════════════

from src.feature_engineering.streaming_ops import (
    decayed_profile, window_distinct, identity_clusters, lag_values,
)


def test_decayed_profile_matches_bruteforce():
    codes, t, v = _entities(200, 8)
    hl = 2 * 86400
    out = decayed_profile(codes, t, v, hl)
    for i in range(len(codes)):
        if codes[i] < 0:
            assert np.isnan(out["count"][i]); continue
        js = [j for j in range(len(codes)) if codes[j] == codes[i] and
              (t[j], j) < (t[i], i) and not np.isnan(v[j])]          # strictly prior rows
        w = np.array([0.5 ** ((t[i] - t[j]) / hl) for j in js])
        assert out["count"][i] == pytest.approx(w.sum(), abs=1e-9)
        if w.sum() > 1e-9:
            assert out["mean"][i] == pytest.approx((w * v[js]).sum() / w.sum(), rel=1e-6)


def test_window_distinct_matches_bruteforce():
    codes, t, _ = _entities(300, 10)
    nb = RNG.integers(-1, 6, len(codes))
    w = 5 * 86400
    out = window_distinct(codes, t, nb, w)
    order = np.lexsort((np.arange(len(t)), t))
    rank = np.empty(len(t), int); rank[order] = np.arange(len(t))
    for i in range(len(codes)):
        if codes[i] < 0:
            assert np.isnan(out[i]); continue
        seen = {nb[j] for j in range(len(codes)) if codes[j] == codes[i] and rank[j] <= rank[i]
                and t[j] > t[i] - w and nb[j] >= 0}
        assert out[i] == len(seen)


def test_lag_values():
    codes = np.array([0, 0, 1, 0, 0])
    t = np.array([1, 2, 3, 4, 5])
    v = np.array([10.0, 20.0, 99.0, 30.0, 40.0])
    np.testing.assert_array_equal(lag_values(codes, t, v, 2), [np.nan, np.nan, np.nan, 10.0, 20.0])


def test_identity_clusters_merge_hubcap_and_delayed_labels():
    day = 86400
    #            client: 0   1   2   0   3   4
    client = np.array([0, 1, 2, 0, 3, 4])
    link = np.array([0, 0, 1, 1, 0, -1])                 # devices
    t = np.array([0, 1, 2, 3, 40, 41]) * day
    newp = np.array([1, 1, 1, 1, 1, 0], dtype=float)
    y = np.array([1, 0, 0, 0, np.nan, 0], dtype=float)
    out = identity_clusters(t, client, link, newp, y, delay_sec=30 * day, hub_cap=2)
    # row 1: client 1 joins device 0 already used by client 0 → merge, reads size 1
    assert out["merge"][1] == 1 and out["size"][1] == 1
    # row 3: client 0 (cluster {0,1}) links device 1 (client 2) → merge of two identities
    assert out["merge"][3] == 1 and out["size"][3] == 2
    # row 4 (day 40): device 0 already has 2 clients = hub cap → no link; client 3 alone
    assert out["size"][4] == 1 and out["merge"][4] == 0
    # labels: none mature before day 30; row 4's own cluster has no fraud
    assert np.all(out["fraud"][:4] == 0)
    with pytest.raises(ValueError):
        identity_clusters(t[::-1], client, link, newp, y, 30 * day)


def test_derived_signals_and_raw_schema(v2_output):
    from src.feature_engineering.pipeline_v2 import run_feature_pipeline_v2
    data, meta, (tr, va, te) = v2_output
    fams = meta["feature_families"]
    for fam in ["derived_signals", "behaviour_v3", "network_v4", "label_history", "client_aggregates"]:
        assert fams.get(fam), f"family {fam} empty"
    feats = set(meta["feature_cols"])
    assert {"amt_gt2_decimals", "addr2_foreign", "P_email_class", "uid_ema_n_1d", "uid_amt_lag2",
            "device_fp_clients_7d", "device_fp_other_clients_lbl_rate", "cluster_n_clients"} <= feats
    # data-study drop list applied
    assert not ({"C3", "D9", "D2", "C10"} & feats)
    raw, rmeta = run_feature_pipeline_v2(tr, va, te, {"schema": "raw", "label_delay_days": 10})
    rfeats = set(rmeta["feature_cols"])
    assert not any(c.startswith("V") and c[1:].isdigit() for c in rfeats)
    assert not any(c in rfeats for c in ["C1", "C14", "D4", "D15"])
    assert "D1" in rfeats and "M4" in rfeats and "TransactionAmt" in rfeats
    assert not any(c.startswith("uid_C") for c in rfeats), "aggregates of vendor columns leaked into raw"


def test_gpu_device_params_translation():
    from src.training.gbdt import apply_device
    base = {"learning_rate": 0.02, "thread_count": -1}
    assert apply_device("xgb", base, "gpu")["device"] == "cuda"
    cb = apply_device("catboost", base, "gpu")
    assert cb["task_type"] == "GPU" and "thread_count" not in cb
    assert apply_device("lgbm", base, "gpu")["device_type"] == "gpu"
    assert apply_device("xgb", base, "cpu") == base               # cpu leaves params untouched
    # PRAUC is not implemented on GPU in CatBoost → GPU-native Logloss for early stopping
    assert apply_device("catboost", {"eval_metric": "PRAUC"}, "gpu")["eval_metric"] == "Logloss"
    assert apply_device("catboost", {"eval_metric": "PRAUC"}, "cpu")["eval_metric"] == "PRAUC"
    # LightGBM OpenCL ids are dropped on CPU
    assert "gpu_platform_id" not in apply_device("lgbm", {"gpu_platform_id": 1}, "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# Phase 7 — online/offline feature PARITY (train/serve skew guard)
# ═════════════════════════════════════════════════════════════════════════════

def test_online_engine_matches_offline_features(v2_output):
    """Replaying every transaction through the online engine must reproduce the
    offline training features (causal mode) for every row and every feature."""
    from src.serving.online_features import OnlineFeatureEngine
    data, meta, (tr, va, te) = v2_output
    off = (pd.concat([data["train"], data["val"], data["test"]], ignore_index=True)
             .sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True))
    raw = pd.concat([tr, va, te], ignore_index=True)
    eng = OnlineFeatureEngine(meta)
    on = eng.replay(raw, labels=True)
    feats = meta["feature_cols"]
    a = off[feats].to_numpy(dtype=np.float64)
    b = on[feats].to_numpy(dtype=np.float64)
    bad = []
    for j, c in enumerate(feats):
        if not np.allclose(a[:, j], b[:, j], rtol=1e-3, atol=1e-3, equal_nan=True):
            k = np.where(~np.isclose(a[:, j], b[:, j], rtol=1e-3, atol=1e-3, equal_nan=True))[0]
            bad.append(f"{c}: {len(k)} rows differ, e.g. row {k[0]} offline={a[k[0], j]} online={b[k[0], j]}")
    assert not bad, "online/offline skew:\n" + "\n".join(bad[:25])


def test_online_single_transaction_after_warmup(v2_output):
    from src.serving.online_features import OnlineFeatureEngine
    data, meta, (tr, va, te) = v2_output
    eng = OnlineFeatureEngine(meta)
    eng.replay(pd.concat([tr, va], ignore_index=True), labels=True, collect=False)
    x = eng.transform(te.iloc[0].to_dict())
    assert list(x.index) == meta["feature_cols"] and np.isfinite(x["TransactionAmt"])
    with pytest.raises(ValueError):
        OnlineFeatureEngine({**meta, "agg_mode": "transductive"})


def test_calibrated_rank_blender(tmp_path):
    from src.training.blending import CalibratedRankBlender, calibration_report
    r = np.random.default_rng(0)
    y = (r.random(20000) < 0.04).astype(int)
    a = 0.6 * y + r.normal(0, 0.3, len(y))            # two differently-scaled models
    b = 100 * (0.5 * y + r.normal(0, 0.3, len(y)))
    bl = CalibratedRankBlender({"a": 0.5, "b": 0.5, "zero": 0.0}).fit({"a": a, "b": b}, y)
    assert "zero" not in bl.weights
    p = bl.predict_proba({"a": a, "b": b})
    assert 0 <= p.min() and p.max() <= 1
    assert abs(p.mean() - y.mean()) < 0.01            # isotonic → calibrated on average
    single = bl.predict_proba({"a": np.array([5.0]), "b": np.array([500.0])})   # one live txn works
    assert single.shape == (1,) and single[0] > 0.5
    bl.save(tmp_path / "b.pkl")
    np.testing.assert_allclose(CalibratedRankBlender.load(tmp_path / "b.pkl").predict_proba({"a": a, "b": b}), p)
    rep = calibration_report(y, p)
    assert rep["ece"] < 0.02 and len(rep["reliability"]) > 3

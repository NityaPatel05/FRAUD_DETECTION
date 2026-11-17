"""Fast single-transaction serving path == pandas/batch path (features, rows, scores).

The dict/numpy row path (src/serving/fast_row.py), the numpy model path (ModelBundle.predict_proba_array) and the
legacy DataFrame path must be interchangeable: same row dict, same feature vector (NaN positions equal,
atol 1e-6), same score. Mutated traffic: missing fields, None/NaN, unseen categories, first-seen clients, odd
floats, numpy scalars, ``id-`` key spelling, exotic types (those fall back to the slow path)."""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.serving.decision import CostPolicy, DecisionLog, DecisionService, ModelBundle
from src.serving.fast_row import FastRowUnsupported
from src.serving.online_features import OnlineFeatureEngine
from test_serving_online import COSTS, POLICY

ROOT = Path(__file__).resolve().parent.parent


def _miss(v):
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def _same_row(fast: dict, slow: dict):
    assert set(fast) == set(slow), f"keys differ: {set(fast) ^ set(slow)}"
    for k, x in fast.items():
        y = slow[k]
        if _miss(x) and _miss(y):
            continue
        assert not _miss(x) and not _miss(y), f"{k}: missing mismatch fast={x!r} slow={y!r}"
        assert type(x).__name__ == type(y).__name__ and x == y, f"{k}: fast={x!r} slow={y!r}"


def _same_vec(a: dict, b: dict):
    ka = list(a)
    assert ka == list(b)
    va, vb = np.array([a[k] for k in ka]), np.array([b[k] for k in ka])
    assert (np.isnan(va) == np.isnan(vb)).all(), [k for k, p, q in zip(ka, va, vb) if np.isnan(p) != np.isnan(q)]
    np.testing.assert_allclose(va, vb, atol=1e-6, rtol=0, equal_nan=True)


# -- traffic generator ----------------------------------------------------------------------------------
_STR_POOL = {
    "P_emaildomain": ["gmail.com", "yahoo.com", "anonymous.com", "protonmail.com", "web.de", "x.mx", "NEW-DOMAIN.io",
                      "Hotmail.COM", "", "a.b.c.fr", "comcast.net", "nodot"],
    "R_emaildomain": ["gmail.com", "icloud.com", "unseen.org", "anonymous.com", "outlook.es"],
    "id_31": ["chrome 63.0", "mobile safari generic", "samsung/sm-g9 browser 4.0", "Safari 11.0.2", "firefox", "",
              "ie 11.0 for desktop", "opera 49.0", "generic/android 7.0", "chrome 100.0.1"],
    "id_30": ["Windows 10", "iOS 11.1.2", "Mac OS X 10_12_6", "Android 7.0", "Linux", "", "windows"],
    "id_33": ["1920x1080", "2560x1440", "x", "800x600", "bad", "1x1"],
    "DeviceInfo": ["SM-G960F Build/R16NW", "Windows", "iOS Device", "rv:11.0", "Hisense-A_1", "", "MacOS"],
    "id_23": ["IP_PROXY:ANONYMOUS", "IP_PROXY:HIDDEN", "IP_PROXY:TRANSPARENT"],
    "id_12": ["Found", "NotFound", "New"], "id_15": ["Found", "New", "Unknown"], "id_16": ["Found", "NotFound"],
    "id_28": ["Found", "New"], "id_29": ["Found", "NotFound"],
    "M1": ["T", "F", "X"], "M4": ["M0", "M1", "M2", "M9"], "ProductCD": ["W", "C", "H", "R", "S", "Z"],
    "card4": ["visa", "mastercard", "discover", "amex", "weird"], "card6": ["debit", "credit", "charge card"],
    "DeviceType": ["desktop", "mobile", "tablet"],
}


def _mutate(base: dict, rng: np.random.Generator, i: int) -> dict:
    t = dict(base)
    keys = [k for k in t if k not in ("TransactionID", "TransactionDT", "isFraud")]
    mode = i % 10
    if mode == 0:                                              # untouched
        return t
    for k in rng.choice(keys, size=int(rng.integers(0, 9)), replace=False):
        if rng.random() < 0.5:
            t.pop(k)                                           # field missing
        else:
            t[k] = None if rng.random() < 0.5 else float("nan")
    for k in keys:                                             # re-draw some values
        if k not in t or rng.random() > 0.25:
            continue
        v = t[k]
        if k in _STR_POOL:
            t[k] = _STR_POOL[k][int(rng.integers(len(_STR_POOL[k])))]
        elif isinstance(v, (float, np.floating)) and not math.isnan(v):
            t[k] = [v * rng.uniform(0.01, 100), round(v, 1), float(rng.integers(0, 500)), 0.1 + 0.2, 87.0, 150.0,
                    np.float32(v), v + 1e-7][int(rng.integers(8))]
        elif isinstance(v, (int, np.integer)):
            t[k] = [int(v) + int(rng.integers(0, 9)), float(v), np.int64(v)][int(rng.integers(3))]
    if mode == 1:                                              # first-seen client / card
        t["card1"] = int(rng.integers(50_000, 99_999))
    if mode == 2 and "TransactionAmt" in t:                    # awkward amounts
        t["TransactionAmt"] = [0.0, 1e-7, 100.0, 99.999, 12.5, 1e6, 0.1 + 0.2, 33.3333, 5.0][int(rng.integers(9))]
    if mode == 3:                                              # Kaggle-test key spelling
        for k in [k for k in list(t) if k.startswith("id_")][:6]:
            t["id-" + k[3:]] = t.pop(k)
    if mode == 4:                                              # exotic types -> slow-path fallback, still equal
        ks = [k for k in ("card2", "addr1", "dist1", "TransactionAmt") if k in t]
        if ks:
            t[ks[0]] = str(t[ks[0]])
    if mode == 5:                                              # id- and id_ both present (bfill merge)
        ks = [k for k in t if k.startswith("id_")]
        if ks:
            t["id-" + ks[0][3:]] = t[ks[0]]
    return t


def _world(schema: str):
    from test_v2_pipeline import _synthetic_ieee
    from src.feature_engineering.pipeline_v2 import run_feature_pipeline_v2
    full = _synthetic_ieee(3000, 21)
    tr, va = full[full.TransactionDT < 86400 * 60], full[full.TransactionDT >= 86400 * 60]
    te = _synthetic_ieee(800, 22, start_id=50_000, labelled=False)
    te["TransactionDT"] += 86400 * 90
    data, meta = run_feature_pipeline_v2(tr, va, te, {"label_policy": POLICY, "schema": schema})
    return data, meta, (tr, va, te)


@pytest.fixture(scope="module", params=["full", "raw"])
def world(request):
    return _world(request.param)


def _traffic(te: pd.DataFrame, n: int, seed: int = 5):
    rng = np.random.default_rng(seed)
    base = te.drop(columns=["isFraud"], errors="ignore").to_dict("records")
    return [_mutate(base[i % len(base)], rng, i) for i in range(n)]


# -- 1. the row dict -------------------------------------------------------------------------------------
def test_fast_row_equals_pandas_row_on_mutated_traffic(world):
    _, meta, (_, _, te) = world
    eng = OnlineFeatureEngine(meta)
    assert eng._fast.ok
    n_fast = 0
    for txn in _traffic(te, 300):
        slow = eng.row_frame(pd.DataFrame([txn])).iloc[0].to_dict()
        try:
            fast = eng._fast.build(txn)
        except FastRowUnsupported:
            continue                                           # engine uses the slow path for these
        n_fast += 1
        _same_row(fast, slow)
    assert n_fast >= 200, "the fast path must cover normal traffic"


@pytest.mark.skipif(not (ROOT / "data/processed/v2_causal_raw/meta.json").exists()
                    or not (ROOT / "data/processed/val_raw.parquet").exists(), reason="real processed data absent")
def test_fast_row_equals_pandas_row_on_real_raw_schema():
    meta = json.loads((ROOT / "data/processed/v2_causal_raw/meta.json").read_text())
    eng = OnlineFeatureEngine(meta)
    val = pd.read_parquet(ROOT / "data/processed/val_raw.parquet").head(300)
    n_fast = 0
    for i, txn in enumerate(_traffic(val, 300, seed=9)):
        slow = eng.row_frame(pd.DataFrame([txn])).iloc[0].to_dict()
        try:
            fast = eng._fast.build(txn)
        except FastRowUnsupported:
            continue
        n_fast += 1
        _same_row(fast, slow)
    assert n_fast >= 200


# -- 2. stateful end to end: features + scores, fast engine vs slow engine -------------------------------
def _bundle(meta, data, tmp: Path) -> ModelBundle:
    import lightgbm as lgb
    from src.training.blending import CalibratedRankBlender
    feats = meta["feature_cols"]
    X = data["train"][feats].astype(np.float32)
    y = data["train"]["isFraud"].fillna(0).to_numpy()
    m = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 31, "seed": 1, "min_data_in_leaf": 5,
                   "deterministic": True}, lgb.Dataset(X, y), 40)
    m.save_model(str(tmp / "lgbm.txt"))
    CalibratedRankBlender({"lgbm": 1.0}).fit({"lgbm": m.predict(X)}, y).save(tmp / "blender.pkl")
    (tmp / "bundle_refit.json").write_text(json.dumps({
        "tag": "t", "features": feats, "cat_cols": [], "model_files": {"lgbm": ["lgbm.txt"]},
        "blender": "blender.pkl", "feature_dtype": "float32"}))
    return ModelBundle(tmp, "refit")


def test_fast_and_slow_engines_give_identical_vectors_and_scores(world, tmp_path):
    data, meta, (tr, va, te) = world
    hist = pd.concat([tr, va], ignore_index=True)
    fast_e, slow_e = OnlineFeatureEngine(meta), OnlineFeatureEngine(meta)
    slow_e.fast_rows = False
    fast_e.replay(hist, labels=True, collect=False)
    slow_e.replay(hist, labels=True, collect=False)
    bundle = _bundle(meta, data, tmp_path)
    bundle_df = ModelBundle(tmp_path, "refit")
    bundle_df.fast_model = False
    fast_s = DecisionService(fast_e, bundle, CostPolicy(COSTS), log=DecisionLog(None))
    slow_s = DecisionService(slow_e, bundle_df, CostPolicy(COSTS), log=DecisionLog(None))
    traffic = _traffic(te, 200, seed=7)
    traffic.sort(key=lambda r: r["TransactionDT"])
    for i, txn in enumerate(traffic):
        txn["TransactionID"] = 900_000 + i                      # unique; mutation never touches it
        rf, vf = fast_e.featurize_vals(dict(txn))
        rs, vs = slow_e.featurize_vals(dict(txn))
        _same_row(rf, rs)
        _same_vec(vf, vs)
        # scoring the identical vector: numpy path == DataFrame path (bit for bit)
        xa = bundle.vector(vf)
        X = pd.Series(vf, dtype=np.float64)[bundle_df.features].to_frame().T
        assert bundle.predict_proba_array(xa)[0] == bundle_df.predict_proba(X)[0]
        assert bundle.reasons_array(xa) == bundle_df.reasons(X)
    # whole service: same decisions through the two routes
    for i, txn in enumerate(_traffic(te, 40, seed=8)):
        txn["TransactionID"] = 950_000 + i
        txn["TransactionDT"] = int(te["TransactionDT"].max()) + 10 + i
        a, b = fast_s.decide(dict(txn)), slow_s.decide(dict(txn))
        for k in ("p_fraud", "action", "decided_by", "reasons", "expected_costs"):
            assert a.get(k) == b.get(k), k


def test_slow_path_flag_and_fallback(world):
    _, meta, (_, _, te) = world
    eng = OnlineFeatureEngine(meta)
    txn = te.drop(columns=["isFraud"], errors="ignore").iloc[0].to_dict()
    r1, _ = eng._row(txn)
    eng.fast_rows = False
    r2, _ = eng._row(txn)
    _same_row(r1, r2)
    eng.fast_rows = True
    r3, _ = eng._row(dict(txn, card1="not-a-number"))           # exotic type -> pandas path, no exception
    assert "uid1" in r3

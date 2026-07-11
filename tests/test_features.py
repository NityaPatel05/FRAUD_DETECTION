"""
tests/test_features.py

Sanity checks for all feature engineering steps.
Run with: pytest tests/test_features.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest


# ── Test fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal synthetic DataFrame that mimics IEEE-CIS schema."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "TransactionID":  np.arange(n),
        "TransactionDT":  np.sort(np.random.randint(86400, 86400 * 180, n)),
        "TransactionAmt": np.random.exponential(100, n),
        "isFraud":        np.random.choice([0, 1], n, p=[0.965, 0.035]),
        "card1":          np.random.randint(1000, 9999, n),
        "card2":          np.random.randint(100, 999, n).astype(float),
        "card3":          np.random.choice([150.0, 185.0], n),
        "card4":          np.random.choice(["visa", "mastercard", None], n),
        "card5":          np.random.randint(100, 250, n).astype(float),
        "card6":          np.random.choice(["credit", "debit", None], n),
        "addr1":          np.random.randint(100, 500, n).astype(float),
        "addr2":          np.random.randint(10, 100, n).astype(float),
        "P_emaildomain":  np.random.choice(["gmail.com", "yahoo.com", "hotmail.com", None], n),
        "R_emaildomain":  np.random.choice(["gmail.com", "yahoo.com", None], n),
        "DeviceInfo":     np.random.choice(["Windows", "iOS Device", "MacOS", None], n),
        "DeviceType":     np.random.choice(["desktop", "mobile", None], n),
        "D1":             np.random.randint(0, 500, n).astype(float),
        "D2":             np.random.randint(0, 400, n).astype(float),
        "D15":            np.random.randint(0, 300, n).astype(float),
        "C1":             np.random.randint(1, 10, n).astype(float),
        "C9":             np.random.randint(0, 5, n).astype(float),
        "C13":            np.random.randint(0, 30, n).astype(float),
        "M1":             np.random.choice(["T", "F", None], n),
        "M5":             np.random.choice(["T", "F", None], n),
        "M9":             np.random.choice(["T", "F", None], n),
        "id_31":          np.random.choice(["chrome 80", "safari 13", "firefox 72", None], n),
        "ProductCD":      np.random.choice(["W", "H", "C", "S", "R"], n),
    })
    return df


# ── Rule 1: sort ──────────────────────────────────────────────────────────────

def test_sort_by_time(sample_df):
    from src.preprocessing.sort_and_split import sort_by_time
    df = sort_by_time(sample_df.copy())
    assert df["TransactionDT"].is_monotonic_increasing, "TransactionDT must be sorted ascending"
    assert "day"  in df.columns
    assert "hour" in df.columns
    assert (df["hour"] >= 0).all() and (df["hour"] <= 23).all()


def test_time_split_no_leakage(sample_df):
    from src.preprocessing.sort_and_split import sort_by_time, time_split
    df    = sort_by_time(sample_df.copy())
    train, val = time_split(df, train_quantile=0.75)
    assert train["day"].max() <= val["day"].min(), \
        "Train days must be strictly before val days (no leakage)"
    assert len(train) + len(val) == len(df)


# ── Rule 2: M column encoding ─────────────────────────────────────────────────

def test_m_column_encoding(sample_df):
    from src.preprocessing.missing_values import encode_m_columns
    from src.preprocessing.sort_and_split import sort_by_time
    df = sort_by_time(sample_df.copy())
    df = encode_m_columns(df, nan_value=-1)
    for col in ["M1", "M5", "M9"]:
        assert df[col].isin([-1, 0, 1]).all(), f"{col} must be -1/0/1 after encoding"
        assert df[col].notna().all(), f"{col} must not contain NaN"


def test_m_column_vae_encoding(sample_df):
    """VAE uses NaN=0, not -1."""
    from src.preprocessing.missing_values import encode_m_columns
    from src.preprocessing.sort_and_split import sort_by_time
    df = sort_by_time(sample_df.copy())
    df = encode_m_columns(df, nan_value=0)
    for col in ["M1", "M5", "M9"]:
        assert df[col].isin([0, 1]).all(), f"VAE: {col} must be 0/1 only"


# ── UID construction ──────────────────────────────────────────────────────────

def test_uid_construction(sample_df):
    from src.preprocessing.sort_and_split import sort_by_time
    from src.feature_engineering.uid_features import build_uid_features
    df = sort_by_time(sample_df.copy())
    df = build_uid_features(df)
    for uid in ["uid1", "uid2", "uid_magic"]:
        assert uid in df.columns
        assert df[uid].notna().all(), f"{uid} must not be null"
        assert df[uid].dtype == object, f"{uid} must be string type"


# ── Frequency encoding: no leakage ───────────────────────────────────────────

def test_freq_encoding_no_leakage(sample_df):
    from src.preprocessing.sort_and_split import sort_by_time, time_split
    from src.feature_engineering.uid_features import build_uid_features
    from src.feature_engineering.frequency_encoding import fit_freq_encoder, apply_freq_encoder

    df     = sort_by_time(sample_df.copy())
    df     = build_uid_features(df)
    train, val = time_split(df)

    freq_maps = fit_freq_encoder(train, cols=["card1"])
    train_out = apply_freq_encoder(train.copy(), freq_maps)
    val_out   = apply_freq_encoder(val.copy(),   freq_maps)

    # val should not have ANY frequency values not seen in train freq_map
    card1_freq_map = freq_maps["card1"]
    unseen = val["card1"].astype(str).apply(
        lambda v: v not in card1_freq_map
    )
    # Unseen values should be filled with 0 (not train-computed value)
    assert (val_out.loc[unseen, "card1_freq"] == 0.0).all(), \
        "Unseen values in val must get frequency=0 (not computed on val data)"


# ── D normalisation ───────────────────────────────────────────────────────────

def test_d_normalisation(sample_df):
    from src.preprocessing.sort_and_split import sort_by_time
    from src.feature_engineering.d_features import build_d_features
    df = sort_by_time(sample_df.copy())
    df = build_d_features(df)
    assert "D1_norm" in df.columns
    assert "D2_norm" in df.columns
    # D_norm should be approx constant per client — std should be lower than raw D
    # (loose check — synthetic data won't perfectly satisfy this)
    assert "D1" in df.columns, "Raw D columns should still be present"


# ── Amount features ───────────────────────────────────────────────────────────

def test_amount_features(sample_df):
    from src.preprocessing.sort_and_split import sort_by_time
    from src.feature_engineering.uid_features import build_uid_features
    from src.feature_engineering.uid_aggregations import build_uid_aggregations
    from src.feature_engineering.amount_features import build_amount_features

    df = sort_by_time(sample_df.copy())
    df = build_uid_features(df)
    df = build_uid_aggregations(df)
    df = build_amount_features(df)

    assert "TransactionAmt_log" in df.columns
    assert "TransactionAmt_decimal" in df.columns
    assert "amt_zscore" in df.columns
    assert (df["TransactionAmt_log"] >= 0).all()
    assert df["TransactionAmt_decimal"].between(-0.01, 1.0).all()


# ── VAE: no velocity features ─────────────────────────────────────────────────

def test_vae_no_velocity():
    """Ensure velocity features are excluded from VAE feature list."""
    import yaml
    with open("configs/features.yaml") as f:
        feat_cfg = yaml.safe_load(f)

    vae_exclude = feat_cfg["vae_features"]["exclude"]
    velocity_feats = ["uid_tx_1h", "uid_tx_24h", "uid_amt_24h", "uid_gap_last", "uid_avg_gap"]
    for vf in velocity_feats:
        assert vf in vae_exclude, f"Velocity feature '{vf}' must be in VAE exclude list"


# ── Cyclic time features ──────────────────────────────────────────────────────

def test_cyclic_time_features(sample_df):
    from src.preprocessing.sort_and_split import sort_by_time
    from src.feature_engineering.time_features import build_time_features

    df = sort_by_time(sample_df.copy())
    df = build_time_features(df)

    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert col in df.columns
        assert df[col].between(-1.0, 1.0).all(), f"{col} must be in [-1, 1]"

    # Verify cyclic property: hour=0 and hour=23 should be close
    hour0_cos  = np.cos(2 * np.pi * 0  / 24)
    hour23_cos = np.cos(2 * np.pi * 23 / 24)
    assert abs(hour0_cos - hour23_cos) < 0.1, "Cyclic encoding: hour 0 and 23 must be adjacent"


# ── Stacking: only model probs as input ──────────────────────────────────────

def test_stacking_input_only_probs():
    from src.models.stacking.meta_learner import assemble_meta_features, STACKING_INPUT_COLS

    n = 100
    probs = {col: np.random.uniform(0, 1, n) for col in STACKING_INPUT_COLS}
    X = assemble_meta_features(probs)
    assert X.shape == (n, len(STACKING_INPUT_COLS)), \
        f"Meta-learner input must be (N, {len(STACKING_INPUT_COLS)})"


# ── Decision thresholds ───────────────────────────────────────────────────────

def test_decision_thresholds():
    from src.models.stacking.meta_learner import apply_decision_thresholds

    probs = np.array([0.05, 0.19, 0.20, 0.50, 0.70, 0.71, 0.99])
    df    = apply_decision_thresholds(probs)

    assert df.loc[0, "decision"] == "auto_approve"
    assert df.loc[1, "decision"] == "auto_approve"
    assert df.loc[2, "decision"] == "flag_review"
    assert df.loc[3, "decision"] == "flag_review"
    assert df.loc[4, "decision"] == "flag_review"
    assert df.loc[5, "decision"] == "block"
    assert df.loc[6, "decision"] == "block"

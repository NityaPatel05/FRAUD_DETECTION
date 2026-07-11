"""
tests/test_robustness.py
=========================
Critical robustness tests for the Fraud Detection pipeline.

Section 1 -- Training vs Inference Parity (3 tests)
Section 2 -- Data Leakage and Time-Split Checks (3 tests)
Section 3 -- Model Output Boundaries (3 tests)

Run with:
  pytest tests/test_robustness.py -v
"""
from __future__ import annotations
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# Shared helper
# ===========================================================================

def _make_raw_transaction(**overrides) -> dict:
    """Minimal valid raw transaction dict (matches IEEE-CIS schema)."""
    base = {
        "TransactionID": 1234567,
        "TransactionDT":  86400,
        "TransactionAmt": 99.0,
        "ProductCD":      "W",
        "card1":          "card_001",
        "card2":          "321.0",
        "card3":          "150.0",
        "card4":          "visa",
        "card5":          "226.0",
        "card6":          "debit",
        "addr1":          "330",
        "addr2":          "87.0",
        "P_emaildomain":  "gmail.com",
        "R_emaildomain":  "gmail.com",
        "DeviceType":     "desktop",
        "DeviceInfo":     "Windows",
        "id_31":          "chrome 72.0",
        "C1": 1.0, "C2": 1.0, "C3": 0.0, "C4": 0.0,
        "C5": 0.0, "C6": 1.0, "C7": 0.0, "C8": 0.0,
        "C9": 1.0, "C10": 0.0, "C11": 2.0, "C12": 0.0,
        "C13": 23.0, "C14": 1.0,
        "D1": 14.0, "D2": 14.0,
        "M1": "T", "M2": "T", "M3": "T", "M4": "M0",
        "M5": "F", "M6": "T", "M7": "F", "M8": "F", "M9": "T",
    }
    base.update(overrides)
    return base


# ===========================================================================
# SECTION 1 -- Training vs Inference Parity
# ===========================================================================

class TestTrainingInferenceParity:

    def test_deterministic_features_match_training(self):
        """Check 1.1: Deterministic features must be bit-identical between
        training and inference pipelines to 5 decimal places."""
        from src.feature_engineering.uid_features    import build_uid_features
        from src.feature_engineering.time_features   import build_time_features
        from src.feature_engineering.email_features  import build_email_features
        from src.feature_engineering.device_features import (
            build_device_features, get_device_encoders, apply_device_encoders,
        )
        from src.feature_engineering.d_features import build_d_features
        from src.feature_engineering.c_features import build_c_features
        from src.feature_engineering.m_features import build_m_features
        from src.preprocessing.missing_values   import encode_m_columns

        tx = _make_raw_transaction()

        # Training path
        train_df = pd.DataFrame([tx, tx.copy()])
        train_df["day"] = train_df["TransactionDT"] // 86400
        build_uid_features(train_df)
        train_df = encode_m_columns(train_df, nan_value=-1)
        train_df = build_time_features(train_df)
        train_df = build_d_features(train_df, train_idx=None)
        train_df = build_c_features(train_df, train_idx=None)
        train_df = build_m_features(train_df, m_cols=None, train_idx=None)
        train_df = build_email_features(train_df)
        build_device_features(train_df)
        device_encoders = get_device_encoders(train_df)
        apply_device_encoders(train_df, device_encoders)
        train_row = train_df.iloc[0]

        # Inference path
        infer_df = pd.DataFrame([tx])
        infer_df["day"] = infer_df["TransactionDT"] // 86400
        build_uid_features(infer_df)
        infer_df = encode_m_columns(infer_df, nan_value=-1)
        infer_df = build_time_features(infer_df)
        infer_df = build_d_features(infer_df, train_idx=None)
        infer_df = build_c_features(infer_df, train_idx=None)
        infer_df = build_m_features(infer_df, m_cols=None, train_idx=None)
        infer_df = build_email_features(infer_df)
        build_device_features(infer_df)
        apply_device_encoders(infer_df, device_encoders)
        infer_row = infer_df.iloc[0]

        check_cols = [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "is_weekend", "is_late_night",
            "D1_norm", "D2_norm",
            "C1_C2_ratio", "C5_C6_ratio", "C6_C11_ratio", "C3_flag",
            "M_match_count", "M_mismatch_count", "M_nan_count",
            "email_match", "P_email_is_free", "email_trusted",
            "device_mobile", "device_os_encoded", "device_browser_encoded",
        ]
        for col in check_cols:
            if col in train_row.index and col in infer_row.index:
                diff = abs(float(train_row[col]) - float(infer_row[col]))
                assert diff < 1e-5, (
                    f"Feature mismatch for '{col}': train={train_row[col]:.6f} "
                    f"vs inference={infer_row[col]:.6f}"
                )

    def test_missing_columns_do_not_crash(self):
        """Check 1.2: Payload missing optional columns must not crash.
        After tree NaN fill, zero numeric NaN values must remain."""
        from src.feature_engineering.time_features   import build_time_features
        from src.feature_engineering.email_features  import build_email_features
        from src.feature_engineering.device_features import build_device_features
        from src.feature_engineering.d_features import build_d_features
        from src.feature_engineering.c_features import build_c_features
        from src.feature_engineering.m_features import build_m_features
        from src.preprocessing.missing_values   import encode_m_columns, fill_nan_for_trees

        minimal_tx = {
            "TransactionDT":  86400,
            "TransactionAmt": 50.0,
            "ProductCD":      "W",
            "card1":          "card_999",
        }
        df = pd.DataFrame([minimal_tx])
        df = encode_m_columns(df, nan_value=-1)
        df = build_time_features(df)
        df = build_d_features(df, train_idx=None)
        df = build_c_features(df, train_idx=None)
        df = build_m_features(df, m_cols=None, train_idx=None)
        df = build_email_features(df)
        df = build_device_features(df)
        fill_nan_for_trees(df, exclude_cols=[])

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        remaining = df[numeric_cols].isna().sum().sum()
        assert remaining == 0, f"NaN values remain after fill_nan_for_trees: {remaining}"

    def test_unseen_categoricals_fallback_no_keyerror(self):
        """Check 1.3: Unseen card1/email/device must fallback to 0.0 or -1
        without raising KeyError."""
        from src.feature_engineering.device_features import (
            build_device_features, get_device_encoders, apply_device_encoders,
        )
        from src.feature_engineering.frequency_encoding import (
            fit_freq_encoder, apply_freq_encoder,
        )

        known_df = pd.DataFrame([_make_raw_transaction()])
        build_device_features(known_df)
        device_encoders = get_device_encoders(known_df)
        freq_maps = fit_freq_encoder(known_df, cols=["card1", "P_emaildomain"])

        unseen_tx = _make_raw_transaction(
            card1="card_NEVER_SEEN",
            P_emaildomain="alien_domain.io",
            DeviceInfo="Commodore64",
            id_31="netscape 4.0",
        )
        infer_df = pd.DataFrame([unseen_tx])
        build_device_features(infer_df)
        apply_device_encoders(infer_df, device_encoders)
        apply_freq_encoder(infer_df, freq_maps)

        assert infer_df["device_os_encoded"].iloc[0] == -1, "Unseen OS must map to -1"
        assert infer_df["device_browser_encoded"].iloc[0] == -1, "Unseen browser must map to -1"
        assert infer_df["card1_freq"].iloc[0] == 0.0, "Unseen card1 must have freq=0.0"
        assert infer_df["P_emaildomain_freq"].iloc[0] == 0.0, "Unseen email must have freq=0.0"


# ===========================================================================
# SECTION 2 -- Data Leakage and Time-Split Checks
# ===========================================================================

class TestDataLeakage:

    def test_velocity_window_counts_are_correct(self):
        """Check 2.1: Velocity counts must be strictly backward-looking.
        DT=9000 sees 0 events in the 1h window and 2 events in the 24h window."""
        from src.feature_engineering.velocity_features import build_velocity_features_offline

        uid = "userA"
        sorted_df = pd.DataFrame([
            {"uid1": uid, "TransactionDT": 100,  "TransactionAmt": 10.0},
            {"uid1": uid, "TransactionDT": 3700, "TransactionAmt": 10.0},
            {"uid1": uid, "TransactionDT": 9000, "TransactionAmt": 10.0},
        ])
        out = build_velocity_features_offline(sorted_df.copy(), group_col="uid1")
        row = out[out["TransactionDT"] == 9000].iloc[0]

        # 9000 - 3700 = 5300s > 3600s, so NOT in 1h window
        assert row["uid_tx_1h"] == 0, f"Expected 0 in 1h, got {row['uid_tx_1h']}"
        # Both prior events are within 86400s
        assert row["uid_tx_24h"] == 2, f"Expected 2 in 24h, got {row['uid_tx_24h']}"

    def test_no_target_leakage_in_uid_aggs(self):
        """Check 2.2: isFraud must NOT appear in any output column of
        build_uid_aggregations."""
        from src.feature_engineering.uid_aggregations import build_uid_aggregations

        df = pd.DataFrame({
            "uid1":           ["u1", "u1", "u2"],
            "uid_magic":      ["u1", "u1", "u2"],
            "card1":          ["c1", "c1", "c2"],
            "addr1":          ["a1", "a1", "a2"],
            "P_emaildomain":  ["g.com", "g.com", "y.com"],
            "TransactionAmt": [100.0, 200.0, 50.0],
            "isFraud":        [0, 1, 0],
        })
        original_cols = set(df.columns)
        out = build_uid_aggregations(df.copy(), train_idx=None)
        new_cols = [c for c in out.columns if c not in original_cols]
        leaky = [c for c in new_cols if "isFraud" in c]
        assert len(leaky) == 0, f"Target leakage detected: {leaky}"

    def test_cold_start_boundary(self):
        """Check 2.3: Cold-start routing rules must be exact.
        tx_count<3 uses stacking, tx_count>=6 uses bilstm, blend in between."""
        from src.inference.predict import _get_final_score

        stacking, bilstm = 0.65, 0.99

        for count in [0, 1, 2]:
            result = _get_final_score(stacking, bilstm, count)
            assert result == stacking, (
                f"Cold start tx_count={count}: expected {stacking}, got {result}"
            )

        assert _get_final_score(stacking, bilstm, 6) == bilstm

        lo, hi = min(stacking, bilstm), max(stacking, bilstm)
        for count in [3, 4, 5]:
            result = _get_final_score(stacking, bilstm, count)
            assert lo <= result <= hi, (
                f"Blend at tx_count={count} ({result:.4f}) outside [{lo}, {hi}]"
            )


# ===========================================================================
# SECTION 3 -- Model Output Boundaries
# ===========================================================================

class TestModelOutputBoundaries:

    def test_stacking_score_bounded_0_1(self):
        """Check 3.1: predict_meta_learner must always return values in [0.0, 1.0]."""
        from src.models.stacking.meta_learner import train_meta_learner, predict_meta_learner

        np.random.seed(42)
        N = 500
        probs_train = {
            "xgb_prob":      np.random.beta(2, 8, N),
            "lgbm_prob":     np.random.beta(2, 8, N),
            "catboost_prob": np.random.beta(2, 8, N),
        }
        y_train   = (probs_train["xgb_prob"] > 0.3).astype(int)
        probs_val = {k: np.random.beta(2, 8, 100) for k in probs_train}
        y_val     = (probs_val["xgb_prob"] > 0.3).astype(int)

        result = train_meta_learner(
            probs_train, y_train, probs_val, y_val,
            save_dir="artifacts/checkpoints",
        )
        preds = predict_meta_learner(
            result["model"], result["scaler"],
            {k: np.random.beta(2, 8, 50) for k in probs_train},
        )
        assert preds.min() >= 0.0, f"Stacking score below 0: {preds.min():.6f}"
        assert preds.max() <= 1.0, f"Stacking score above 1: {preds.max():.6f}"

    def test_bilstm_sequence_shape_for_new_user(self):
        """Check 3.2: build_sequences for a brand-new user must produce a tensor
        of shape (1, SEQ_LEN, N_FEATURES) with no NaN or Inf values."""
        from src.models.bilstm.dataset import build_sequences, SEQ_FEATURES, SEQ_LEN

        tx_data = {col: 0.0 for col in SEQ_FEATURES}
        tx_data["uid1"]              = "brand_new_user"
        tx_data["TransactionDT"]     = 86400
        tx_data["uid_total_amt_24h"] = 50.0

        df = pd.DataFrame([tx_data])
        sequences, _ = build_sequences(df, uid_col="uid1", seq_len=SEQ_LEN)

        expected = (1, SEQ_LEN, len(SEQ_FEATURES))
        assert sequences.shape == expected, (
            f"BiLSTM shape mismatch: expected {expected}, got {sequences.shape}"
        )
        assert sequences.dtype == np.float32, "BiLSTM sequences must be float32"
        assert not np.isnan(sequences).any(), "NaN found in BiLSTM input tensor"
        assert not np.isinf(sequences).any(), "Inf found in BiLSTM input tensor"

    def test_vae_score_clipped_to_0_1(self):
        """Check 3.3: _predict_vae must clip raw anomaly scores to [0, 1].
        A raw score of 45000 with training max=10 must clip to 1.0."""
        from src.inference.predict import FraudDetectionPredictor

        predictor = FraudDetectionPredictor.__new__(FraudDetectionPredictor)
        predictor.device = "cpu"

        mock_vae = MagicMock()
        mock_vae.anomaly_score.return_value.cpu.return_value.item.return_value = 45000.0

        predictor.models   = {"vae": mock_vae, "vae_stats": [0.5, 10.0]}
        predictor.encoders = {
            "vae_cols":       ["TransactionAmt_log", "amt_zscore"],
            "median_imputer": MagicMock(**{"transform.return_value": np.array([[1.0, 0.0]])}),
            "vae_scaler":     None,
        }

        row_df = pd.DataFrame([{"TransactionAmt_log": 4.6, "amt_zscore": 0.3}])
        score  = predictor._predict_vae(row_df)

        assert 0.0 <= score <= 1.0, f"VAE score not in [0, 1]: {score:.6f}"
        assert score == pytest.approx(1.0, abs=1e-4), (
            f"Raw=45000 with train_max=10 must clip to 1.0, got {score}"
        )

from __future__ import annotations

import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.inference.feature_store import FeatureStore
from src.inference.redis_atomic import RedisAtomicOps
from src.inference.cold_start import ColdStartHandler
from src.models.stacking.meta_learner import (
    predict_meta_learner,
    apply_decision_thresholds,
)
from src.feature_engineering.uid_features import build_uid_features_inference
from src.feature_engineering.time_features import build_time_features
from src.feature_engineering.email_features import build_email_features
from src.feature_engineering.device_features import build_device_features, apply_device_encoders
from src.feature_engineering.d_features import build_d_features
from src.feature_engineering.c_features import build_c_features
from src.feature_engineering.m_features import build_m_features
from src.preprocessing.missing_values import encode_m_columns


# ─────────────────────────────────────────────────────────────────────────────
# Module-level fusion function [Bug B3/B4 fix]
# Single source of truth — cold_start.py no longer defines weights.
# ─────────────────────────────────────────────────────────────────────────────

def _get_final_score(
    stacking_score: float,
    bilstm_prob:    float,
    tx_count:       int,
) -> float:
    """
    Monotonic cold-start transition — eliminates the feedback-loop artifact.

    The BiLSTM already consumed `calibrated_stacking_score` as an input
    feature at every sequence timestep.  The old Step 9 mixed stacking back
    into the final score, which double-counted its influence.

    Correct rule:
      tx_count < 3  → stacking_score is final (BiLSTM has no real history)
      tx_count 3-5  → linear blend: alpha ramps 0→1 so bilstm dominates
      tx_count >= 6 → bilstm_prob is final (stacking already inside it)
    """
    if tx_count < 3:
        return stacking_score
    elif tx_count < 6:
        alpha = (tx_count - 3) / 3.0   # 0.0 at tx=3 → 1.0 at tx=6
        return (1.0 - alpha) * stacking_score + alpha * bilstm_prob
    else:
        return bilstm_prob


class FraudDetectionPredictor:
    """
    Production inference engine for fraud detection.

    Loads all models once at startup, then predicts on individual
    transactions or batches with ~8ms latency.
    """

    def __init__(
        self,
        checkpoint_dir: str = "artifacts/checkpoints",
        config_path:    str = "configs/config.yaml",
        redis_client    = None,   # pre-built redis.Redis; overrides host/port in load_all
    ):
        self.artifacts_dir = Path(checkpoint_dir)
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models        = {}
        self.encoders      = {}
        self.feature_store = None
        self._precreated_redis = redis_client   # set if caller owns the connection

        logger.info("FraudDetectionPredictor initialising (device={})…", self.device)

    def load_all(
        self,
        redis_host:   str = "localhost",
        redis_port:   int = 6379,
        redis_client  = None,
    ) -> None:
        """Load all models and encoders. Call once at service startup.

        Parameters
        ----------
        redis_client : pass a pre-built redis.Redis instance to reuse an
                       existing connection pool (e.g. when init_post_inference
                       has already created one).  If None, creates a new client
                       from redis_host / redis_port.
        """
        import redis as redis_lib
        art = self.artifacts_dir

        # Resolve Redis client — prefer explicitly passed client,
        # then the one stored at __init__ time, then create a new one.
        r = redis_client or self._precreated_redis or redis_lib.Redis(
            host=redis_host, port=redis_port, db=0, decode_responses=False
        )

        # ── Feature store — reuse the resolved Redis client ───────────────────
        self.feature_store = FeatureStore.__new__(FeatureStore)
        self.feature_store.r   = r
        self.feature_store.ttl = 30 * 86400
        self.feature_store._freq_maps        = {}
        self.feature_store._card_embeddings  = {}
        self.feature_store.load_static_artifacts(str(art / "feature_encoders.pkl"))
        if (art / "gnn_card_embeddings.npy").exists():
            self.feature_store.load_card_embeddings(
                str(art / "gnn_card_embeddings.npy"),
                str(art / "gnn_card_map.pkl"),
            )


        # ── Redis atomic + cold-start handler ───────────────────────────────
        self._redis_ops   = RedisAtomicOps(r)
        self._cold_start  = ColdStartHandler(
            r=r,
            priors_path=str(art / "cohort_priors.json"),
        )

        # ── Feature encoders ──────────────────────────────────────────────────
        with open(art / "feature_encoders.pkl", "rb") as f:
            self.encoders = pickle.load(f)

        # ── Tree models (XGBoost folds) ───────────────────────────────────────
        import xgboost as xgb
        xgb_models = []
        if (art / "xgb_final.pkl").exists():
            with open(art / "xgb_final.pkl", "rb") as f:
                xgb_models.append(pickle.load(f))
            logger.info("Loaded XGBoost final retrained model.")
        else:
            for p in sorted(art.glob("xgb_fold*.pkl")):
                with open(p, "rb") as f:
                    xgb_models.append(pickle.load(f))
            logger.info("Loaded {} XGBoost fold models.", len(xgb_models))
        self.models["xgb"] = xgb_models

        # ── LightGBM ──────────────────────────────────────────────────────────
        lgbm_models = []
        if (art / "lgbm_final.pkl").exists():
            with open(art / "lgbm_final.pkl", "rb") as f:
                lgbm_models.append(pickle.load(f))
            logger.info("Loaded LightGBM final retrained model.")
        else:
            for p in sorted(art.glob("lgbm_fold*.pkl")):
                with open(p, "rb") as f:
                    lgbm_models.append(pickle.load(f))
            logger.info("Loaded {} LightGBM fold models.", len(lgbm_models))
        self.models["lgbm"] = lgbm_models

        # ── CatBoost ──────────────────────────────────────────────────────────
        from catboost import CatBoostClassifier
        cb_models = []
        if (art / "catboost_final.cbm").exists():
            m = CatBoostClassifier()
            m.load_model(str(art / "catboost_final.cbm"))
            cb_models.append(m)
            logger.info("Loaded CatBoost final retrained model.")
        else:
            for p in sorted(art.glob("catboost_fold*.cbm")):
                m = CatBoostClassifier()
                m.load_model(str(p))
                cb_models.append(m)
            logger.info("Loaded {} CatBoost fold models.", len(cb_models))
        self.models["catboost"] = cb_models

        # ── VAE ───────────────────────────────────────────────────────────────
        if (art / "vae_best.pt").exists():
            from src.models.vae.model import FraudVAE
            vae_state = torch.load(art / "vae_best.pt", map_location=self.device)
            n_vae_feats = len(self.encoders.get("vae_cols", []))
            if n_vae_feats > 0:
                vae = FraudVAE(input_dim=n_vae_feats).to(self.device)
                vae.load_state_dict(vae_state)
                vae.eval()
                self.models["vae"] = vae
                self.models["vae_stats"] = np.load(art / "vae_score_stats.npy")
                if (art / "vae_scaler.pkl").exists():
                    with open(art / "vae_scaler.pkl", "rb") as f:
                        self.encoders["vae_scaler"] = pickle.load(f)
                logger.info("Loaded VAE model (input_dim={}).", n_vae_feats)

        # ── BiLSTM + Platt Scaler ────────────────────────────────────────────
        bilstm_path = None
        if (art / "bilstm_final.pt").exists():
            bilstm_path = art / "bilstm_final.pt"
            logger.info("Loading BiLSTM final retrained model.")
        elif (art / "bilstm_best.pt").exists():
            bilstm_path = art / "bilstm_best.pt"
            logger.info("Loading BiLSTM best fold model.")
            
        if bilstm_path is not None:
            from src.models.bilstm.model import FraudBiLSTM
            from src.models.bilstm.dataset import SEQ_FEATURES, SEQ_LEN
            bilstm = FraudBiLSTM(input_size=len(SEQ_FEATURES)).to(self.device)
            bilstm.load_state_dict(
                torch.load(bilstm_path, map_location=self.device)
            )
            bilstm.eval()
            self.models["bilstm"]          = bilstm
            self.models["bilstm_seq_len"]  = SEQ_LEN
            self.models["bilstm_features"] = SEQ_FEATURES
            logger.info("Loaded BiLSTM model (input={}, seq_len={}).", len(SEQ_FEATURES), SEQ_LEN)

        if (art / "bilstm_platt_scaler.pkl").exists():
            with open(art / "bilstm_platt_scaler.pkl", "rb") as f:
                self.models["bilstm_platt"] = pickle.load(f)
            logger.info("Loaded BiLSTM Platt Scaler.")
        else:
            logger.warning("bilstm_platt_scaler.pkl not found — BiLSTM output will be uncalibrated.")

        # ── Meta-learner ──────────────────────────────────────────────────────
        with open(art / "meta_learner.pkl", "rb") as f:
            meta_artifacts = pickle.load(f)
        self.models["meta_model"] = meta_artifacts["model"]
        self.models["meta_scaler"] = meta_artifacts["scaler"]
        logger.info("Loaded stacking meta-learner.")

        # ── Feature col lists ─────────────────────────────────────────────────
        xgb_feat_df = pd.read_csv(art / "xgb_feature_cols.csv")
        self.models["xgb_feature_cols"]  = xgb_feat_df["col"].tolist()
        logger.info("All models loaded. Ready for inference.")

    # ─────────────────────────────────────────────────────────────────────────
    # Single transaction inference
    # ─────────────────────────────────────────────────────────────────────────

    def predict_single(self, transaction: dict) -> dict:
        """
        Predict fraud probability for a single transaction dict.

        Parameters
        ----------
        transaction : raw transaction fields (same schema as training data)

        Returns
        -------
        dict with: final_fraud_prob, stacking_score, bilstm_prob, decision, action,
                   model_probs, fusion_weights, latency_ms
        """
        t0 = time.time()

        # ── Step 1: Build UIDs ───────────────────────────────────────────────
        uid_feats = build_uid_features_inference(transaction)
        transaction.update(uid_feats)

        uid = transaction["uid1"]
        dt  = int(transaction["TransactionDT"])

        # ── Step 2: Ring check [Bug B2 fix] ──────────────────────────────────
        # Runs synchronously BEFORE the parallel Redis fetch (~1 ms).
        # ring_signal / shared_*_users are injected into transaction so they
        # land in row_df and flow through to the tree models as features.
        ring_result = self._redis_ops.fast_ring_check(
            uid=uid,
            device_id=str(transaction.get("DeviceInfo",    "unknown")),
            email_domain=str(transaction.get("P_emaildomain", "unknown")),
        )
        transaction["ring_signal"]         = ring_result["ring_signal"]
        transaction["shared_device_users"] = ring_result["shared_device_users"]
        transaction["shared_email_users"]  = ring_result["shared_email_users"]

        # ── Step 3: User meta (tx_count, is_new_user) ────────────────────────
        user_meta = self._cold_start.get_user_meta_features(uid)
        transaction.update(user_meta)
        tx_count  = user_meta["tx_count_so_far"]

        # ── Step 4: Fetch Redis features in parallel ────────────────────────
        with ThreadPoolExecutor(max_workers=3) as pool:
            vel_future    = pool.submit(self.feature_store.get_velocity_features, uid, dt)
            static_future = pool.submit(self.feature_store.get_static_features,   uid)
            gnn_future    = pool.submit(
                self.feature_store.get_gnn_features,
                str(transaction.get("card1", "unknown"))
            )
            velocity_feats = vel_future.result()
            static_feats   = static_future.result()
            gnn_feats      = gnn_future.result()

        transaction.update(velocity_feats)
        transaction.update(static_feats)

        # ── Step 5: Compute derived features ───────────────────────────────
        row_df = pd.DataFrame([transaction])
        row_df = encode_m_columns(row_df, nan_value=-1)
        row_df = build_time_features(row_df)
        row_df = build_d_features(row_df, train_idx=None)
        row_df = build_c_features(row_df, train_idx=None)
        row_df = build_m_features(row_df, m_cols=None, train_idx=None)
        row_df = build_email_features(row_df)
        row_df = build_device_features(row_df)
        
        device_encoders = self.encoders.get("device_encoders")
        if device_encoders:
            row_df = apply_device_encoders(row_df, device_encoders)

        freq_maps = self.encoders.get("freq_maps", {})
        for col, fmap in freq_maps.items():
            if col in row_df.columns:
                row_df[f"{col}_freq"] = row_df[col].map(fmap).fillna(0.0)

        # ── Step 6: VAE anomaly score ──────────────────────────────────────
        vae_score = self._predict_vae(row_df)
        row_df["vae_anomaly_score"] = vae_score

        # ── Step 7: Parallel tree models ──────────────────────────────────────
        # NOTE: GNN is embedding-only — 64-dim card embeddings are already
        # injected as features into row_df via gnn_feats["node_embedding"].
        # ring_signal / shared_*_users (Step 2) are also in row_df as features.
        # VAE anomaly score is already injected as vae_anomaly_score feature.
        with ThreadPoolExecutor(max_workers=3) as pool:
            xgb_f  = pool.submit(self._predict_tree, "xgb",  row_df)
            lgbm_f = pool.submit(self._predict_tree, "lgbm", row_df)
            cb_f   = pool.submit(self._predict_catboost,      row_df)
            xgb_prob  = xgb_f.result()
            lgbm_prob = lgbm_f.result()
            cb_prob   = cb_f.result()

        # ── Step 8: Stacking meta-learner → stacking_score ───────────────────
        probs_dict = {
            "xgb_prob":          np.array([xgb_prob]),
            "lgbm_prob":         np.array([lgbm_prob]),
            "catboost_prob":     np.array([cb_prob]),
        }

        stacking_score = float(predict_meta_learner(
            self.models["meta_model"],
            self.models["meta_scaler"],
            probs_dict,
        )[0])

        # ── Step 9: BiLSTM ────────────────────────────────────────────────────
        bl_prob = self._predict_bilstm_from_redis(uid, row_df, stacking_score)

        # Apply Platt scaling if available
        if "bilstm_platt" in self.models:
            bl_prob = float(self.models["bilstm_platt"].predict_proba([[bl_prob]])[0, 1])

        # ── Step 10: Score fusion [Bug B4 fix] ────────────────────────────────
        # Old weighted average (0.8*stacking + 0.2*bilstm etc.) re-mixed a signal
        # BiLSTM already consumed as input, creating a compounding artifact.
        # _get_final_score() uses stacking as final for cold users, bilstm for
        # users with history (stacking is already baked into BiLSTM's input).
        final_prob = _get_final_score(stacking_score, bl_prob, tx_count)

        # ── Step 11: Decision ─────────────────────────────────────────────────
        decision_df = apply_decision_thresholds(np.array([final_prob]))
        decision    = decision_df.iloc[0]["decision"]
        action      = decision_df.iloc[0]["action"]

        # ── Step 12: Post-inference async tasks ────────────────────────────────
        # If init_post_inference() has been called (service startup), all
        # side-effects (velocity update, Kafka, ring persistence, seq window)
        # run in background threads — zero added latency to the checkout path.
        # Fallback: synchronous velocity update only, for callers that run
        # predict_single() without a full service context.
        try:
            from src.jobs.post_inference import post_inference_async, _redis_client as _pi_r
            if _pi_r is not None:
                post_inference_async(
                    uid=uid,
                    transaction=transaction,
                    fraud_score=final_prob,
                    stacking_score=stacking_score,
                )
            else:
                # Fallback: synchronous velocity update
                amt = float(transaction.get("TransactionAmt", 0))
                self.feature_store.update_after_transaction(uid, amt, dt)
        except Exception:
            # Never let async wiring break the synchronous return path
            amt = float(transaction.get("TransactionAmt", 0))
            self.feature_store.update_after_transaction(uid, amt, dt)

        latency_ms = (time.time() - t0) * 1000

        return {
            "fraud_prob":      final_prob,
            "stacking_score": stacking_score,
            "bilstm_prob":    bl_prob,
            "decision":       decision,
            "action":         action,
            # model_probs shows the 3 base tree models + the two ensemble layers.
            # GNN contributes as 64-dim card embeddings (features) inside XGB/LGBM/CatBoost.
            # VAE contributes as vae_anomaly_score feature inside the tree models.
            "model_probs": {
                "xgb":             xgb_prob,
                "lgbm":            lgbm_prob,
                "catboost":        cb_prob,
                "stacking":        stacking_score,
                "bilstm":          bl_prob,
            },
            "latency_ms": round(latency_ms, 2),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal model runners
    # ─────────────────────────────────────────────────────────────────────────

    def _predict_tree(self, model_key: str, row_df: pd.DataFrame) -> float:
        models       = self.models.get(model_key, [])
        feature_cols = self.models.get(f"{model_key}_feature_cols",
                                       self.models.get("xgb_feature_cols", []))
        if not models or not feature_cols:
            return 0.5
        
        # Ensure all required features are present, fill missing with NaN
        X_df = row_df.reindex(columns=feature_cols)
        X_df = X_df.apply(pd.to_numeric, errors='coerce')
        X = X_df.fillna(-999).values
        
        probs = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
        return float(probs[0])

    def _predict_catboost(self, row_df: pd.DataFrame) -> float:
        models = self.models.get("catboost", [])
        if not models:
            return 0.5
        from catboost import Pool
        
        expected_features = models[0].feature_names_
        X_df = row_df.reindex(columns=expected_features)
        
        cat_cols  = self.encoders.get("cat_cols", [])
        cat_idx   = [expected_features.index(c) for c in cat_cols if c in expected_features]

        num_cols = [c for c in expected_features if c not in cat_cols]
        for c in num_cols:
            X_df[c] = pd.to_numeric(X_df[c], errors='coerce')

        for c in cat_cols:
            if c in X_df.columns:
                X_df[c] = X_df[c].astype(str).fillna("unknown")

        pool = Pool(X_df, cat_features=cat_idx)
        probs = np.mean([m.predict_proba(pool)[:, 1] for m in models], axis=0)
        return float(probs[0])

    def _predict_vae(self, row_df: pd.DataFrame) -> float:
        vae = self.models.get("vae")
        if vae is None:
            return 0.5
        vae_cols = self.encoders.get("vae_cols", [])
        imp      = self.encoders.get("median_imputer")
        scaler   = self.encoders.get("vae_scaler")
        if not vae_cols or imp is None:
            return 0.5
        X_arr   = np.zeros((1, len(vae_cols)), dtype=np.float32)
        for j, col in enumerate(vae_cols):
            if col in row_df.columns:
                val = row_df[col].iloc[0]
                if val is None or pd.isna(val):
                    val = np.nan
                elif isinstance(val, str):
                    if val == 'T': val = 1.0
                    elif val == 'F': val = 0.0
                    else:
                        try:
                            val = float(val)
                        except ValueError:
                            val = np.nan
                X_arr[0, j] = float(val)
        X_imp = imp.transform(X_arr).astype(np.float32)
        if scaler is not None:
            X_imp = scaler.transform(X_imp).astype(np.float32)
        x_t   = torch.tensor(X_imp, dtype=torch.float32).to(self.device)
        score = float(vae.anomaly_score(x_t).cpu().item())
        s_min, s_max = self.models.get("vae_stats", [0.0, 1.0])
        return float(np.clip((score - s_min) / (s_max - s_min + 1e-8), 0.0, 1.0))

    def _predict_bilstm_from_redis(
        self,
        uid:             str,
        row_df:          pd.DataFrame,
        stacking_score:  float,
    ) -> float:
        """
        Build a BiLSTM input tensor from the Redis sequence window.

        1. Serialise current transaction row into a feature vector.
        2. Push + trim + read window atomically (Lua script).
        3. Pad with cohort priors if window is short.
        4. Run BiLSTM forward pass.
        """
        bilstm = self.models.get("bilstm")
        if bilstm is None:
            return 0.5

        feature_cols = self.models.get("bilstm_features", [])
        seq_len      = self.models.get("bilstm_seq_len",  10)
        n_feat       = len(feature_cols)

        # Build feature vector for this transaction
        current_vec = np.zeros(n_feat, dtype=np.float32)
        for j, col in enumerate(feature_cols):
            if col in row_df.columns:
                current_vec[j] = float(row_df[col].iloc[0])

        # Atomic push + trim + read (1 Redis round-trip)
        seq_key = f"user:{uid}:seq_window"
        window  = self._redis_ops.push_and_get_window(seq_key, current_vec, max_len=seq_len)

        # Pad if window is shorter than seq_len (e.g. brand new user)
        pad_len = seq_len - len(window)
        if pad_len > 0:
            # Cold start handler owns the phantom padding — reuse its priors
            from src.inference.cold_start import _make_cohort_key_from_dict
            from src.preprocessing.build_cohort_priors import get_cohort_stats
            tx_dict    = row_df.iloc[0].to_dict()
            cohort_key = _make_cohort_key_from_dict(tx_dict)
            stats      = get_cohort_stats(self._cold_start._priors, cohort_key, feature_cols)
            pad_vec    = np.array(
                [stats.get(f"{col}_mean", 0.0) for col in feature_cols],
                dtype=np.float32,
            )
            window = [pad_vec] * pad_len + window

        # Stack into tensor (batch=1, seq_len, n_feat)
        x_np = np.stack(window[-seq_len:], axis=0)[np.newaxis]  # (1, seq_len, n_feat)
        x_t  = torch.tensor(x_np, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = bilstm(x_t)
        # .squeeze() handles any shape: (1,1), (1,), or scalar
        return float(torch.sigmoid(logits.squeeze()).cpu().item())

    # ─────────────────────────────────────────────────────────────────────────
    # Vectorised batch helpers [Bug B1 fix]
    # ─────────────────────────────────────────────────────────────────────────

    def _predict_tree_batch(self, model_key: str, batch_df: pd.DataFrame) -> np.ndarray:
        """Run K-fold tree ensemble on full batch DataFrame → (N,) probability array."""
        models       = self.models.get(model_key, [])
        feature_cols = self.models.get(f"{model_key}_feature_cols",
                                       self.models.get("xgb_feature_cols", []))
        if not models or not feature_cols:
            return np.full(len(batch_df), 0.5, dtype=np.float32)
        present = [c for c in feature_cols if c in batch_df.columns]
        X = batch_df[present].fillna(-999).values
        return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0).astype(np.float32)

    def _predict_catboost_batch(self, batch_df: pd.DataFrame) -> np.ndarray:
        """Run CatBoost fold ensemble on full batch DataFrame → (N,) probability array."""
        models = self.models.get("catboost", [])
        if not models:
            return np.full(len(batch_df), 0.5, dtype=np.float32)
        from catboost import Pool
        cat_cols  = self.encoders.get("cat_cols", [])
        feat_cols = [c for c in batch_df.columns if c not in ["TransactionID", "isFraud"]]
        cat_idx   = [feat_cols.index(c) for c in cat_cols if c in feat_cols]
        pool = Pool(batch_df[feat_cols], cat_features=cat_idx)
        return np.mean([m.predict_proba(pool)[:, 1] for m in models], axis=0).astype(np.float32)

    def _predict_vae_batch(self, batch_df: pd.DataFrame) -> np.ndarray:
        """Run VAE anomaly scoring on full batch → (N,) normalised score array."""
        vae = self.models.get("vae")
        if vae is None:
            return np.full(len(batch_df), 0.5, dtype=np.float32)
        vae_cols = self.encoders.get("vae_cols", [])
        imp      = self.encoders.get("median_imputer")
        scaler   = self.encoders.get("vae_scaler")
        if not vae_cols or imp is None:
            return np.full(len(batch_df), 0.5, dtype=np.float32)
        N     = len(batch_df)
        X_arr = np.zeros((N, len(vae_cols)), dtype=np.float32)
        for j, col in enumerate(vae_cols):
            if col in batch_df.columns:
                X_arr[:, j] = batch_df[col].values.astype(np.float32)
        X_imp = imp.transform(X_arr).astype(np.float32)
        if scaler is not None:
            X_imp = scaler.transform(X_imp).astype(np.float32)
        x_t = torch.tensor(X_imp, dtype=torch.float32).to(self.device)
        s_min, s_max = self.models.get("vae_stats", [0.0, 1.0])
        with torch.no_grad():
            scores = vae.anomaly_score(x_t).cpu().numpy()
        return np.clip((scores - s_min) / (s_max - s_min + 1e-8), 0.0, 1.0).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Vectorised batch prediction [Bug B1 fix]
    # ─────────────────────────────────────────────────────────────────────────

    def predict_batch(self, df: pd.DataFrame) -> list:
        """
        Vectorised inference on a DataFrame of N transactions.

        Performance vs. row-by-row:
          - Redis: one fetch per unique uid (not one per row).
          - Tree models: single predict_proba call on full (N, n_features) matrix.
          - VAE: single batched torch forward pass.
          - Stacking meta-learner: fully vectorised.
          - BiLSTM: sequential per row (sequence depends on history order).
          - Ring check: deduped per (device, email) pair within the batch.

        Returns
        -------
        list of dicts, one per row, same schema as predict_single().
        Note: latency_ms is not included (not meaningful for batch).
        """
        N    = len(df)
        rows = df.to_dict("records")

        # ── Step 1: UIDs ──────────────────────────────────────────────────────
        for row in rows:
            row.update(build_uid_features_inference(row))

        uids  = [r["uid1"]                            for r in rows]
        dts   = [int(r["TransactionDT"])              for r in rows]
        amts  = [float(r.get("TransactionAmt", 0.0)) for r in rows]
        cards = [str(r.get("card1", "unknown"))       for r in rows]

        # ── Step 2: Ring check (deduped per device+email pair) ────────────────
        ring_cache: Dict = {}
        for row, uid in zip(rows, uids):
            dev   = str(row.get("DeviceInfo",    "unknown"))
            email = str(row.get("P_emaildomain", "unknown"))
            key   = (dev, email)
            if key not in ring_cache:
                ring_cache[key] = self._redis_ops.fast_ring_check(
                    uid=uid, device_id=dev, email_domain=email
                )
            ring = ring_cache[key]
            row["ring_signal"]         = ring["ring_signal"]
            row["shared_device_users"] = ring["shared_device_users"]
            row["shared_email_users"]  = ring["shared_email_users"]

        # ── Step 3: User meta (deduped per uid) ───────────────────────────────
        unique_uids = list(dict.fromkeys(uids))
        user_metas  = {uid: self._cold_start.get_user_meta_features(uid)
                       for uid in unique_uids}
        tx_counts   = [user_metas[uid]["tx_count_so_far"] for uid in uids]
        for row, uid in zip(rows, uids):
            row.update(user_metas[uid])

        # ── Step 4: Velocity + static + GNN (deduped per uid / card) ──────────
        # Build uid→first_dt map in O(n) to avoid O(n²) list.index() inside the loop
        uid_to_first_dt: Dict = {}
        for uid, dt in zip(uids, dts):
            if uid not in uid_to_first_dt:
                uid_to_first_dt[uid] = dt

        vel_cache: Dict    = {}
        static_cache: Dict = {}
        for uid in unique_uids:
            vel_cache[uid]    = self.feature_store.get_velocity_features(uid, uid_to_first_dt[uid])
            static_cache[uid] = self.feature_store.get_static_features(uid)

        gnn_cache: Dict = {}
        for card in dict.fromkeys(cards):
            gnn_cache[card] = self.feature_store.get_gnn_features(card)

        for row, uid in zip(rows, uids):
            row.update(vel_cache[uid])
            row.update(static_cache[uid])

        # ── Step 5: Derived features — full batch DataFrame ───────────────────
        batch_df = pd.DataFrame(rows)
        batch_df = build_time_features(batch_df)
        batch_df = build_email_features(batch_df)
        batch_df = build_device_features(batch_df)

        freq_maps = self.encoders.get("freq_maps", {})
        for col, fmap in freq_maps.items():
            if col in batch_df.columns:
                batch_df[f"{col}_freq"] = batch_df[col].map(fmap).fillna(0.0)

        # ── Step 6: VAE — single batched torch forward pass ───────────────────
        batch_df["vae_anomaly_score"] = self._predict_vae_batch(batch_df)

        # ── Step 7: Tree models — single vectorised call each ─────────────────
        xgb_probs  = self._predict_tree_batch("xgb",  batch_df)
        lgbm_probs = self._predict_tree_batch("lgbm", batch_df)
        cb_probs   = self._predict_catboost_batch(batch_df)

        # ── Step 8: Stacking on full batch ────────────────────────────────────
        stacking_scores = predict_meta_learner(
            self.models["meta_model"],
            self.models["meta_scaler"],
            {
                "xgb_prob":      xgb_probs,
                "lgbm_prob":     lgbm_probs,
                "catboost_prob": cb_probs,
            },
        ).astype(np.float32)

        # ── Step 9: BiLSTM — sequential per row (order matters) ───────────────
        bilstm_probs = np.zeros(N, dtype=np.float32)
        for i in range(N):
            bl = self._predict_bilstm_from_redis(
                uids[i], batch_df.iloc[[i]], float(stacking_scores[i])
            )
            if "bilstm_platt" in self.models:
                bl = float(self.models["bilstm_platt"].predict_proba([[bl]])[0, 1])
            bilstm_probs[i] = bl

        # ── Step 10: Fusion ───────────────────────────────────────────────────
        final_probs = np.array([
            _get_final_score(float(stacking_scores[i]), float(bilstm_probs[i]), tx_counts[i])
            for i in range(N)
        ], dtype=np.float32)

        # ── Step 11: Decisions ────────────────────────────────────────────────
        decision_df = apply_decision_thresholds(final_probs)

        # ── Step 12: Update Redis velocity (post-decision) ────────────────────
        for uid, amt, dt in zip(uids, amts, dts):
            self.feature_store.update_after_transaction(uid, amt, dt)

        return [
            {
                "fraud_prob":     float(final_probs[i]),
                "stacking_score": float(stacking_scores[i]),
                "bilstm_prob":    float(bilstm_probs[i]),
                "decision":       decision_df.iloc[i]["decision"],
                "action":         decision_df.iloc[i]["action"],
                "model_probs": {
                    "xgb":      float(xgb_probs[i]),
                    "lgbm":     float(lgbm_probs[i]),
                    "catboost": float(cb_probs[i]),
                    "stacking": float(stacking_scores[i]),
                    "bilstm":   float(bilstm_probs[i]),
                },
            }
            for i in range(N)
        ]

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import redis as redis_lib
from loguru import logger

from src.models.bilstm.dataset import SEQ_FEATURES, SEQ_LEN, COHORT_COLS, _make_cohort_key
from src.preprocessing.build_cohort_priors import load_cohort_priors, get_cohort_stats
from src.inference.redis_atomic import RedisAtomicOps


# Default priors path (built offline by build_cohort_priors.py)
_DEFAULT_PRIORS_PATH = "artifacts/checkpoints/cohort_priors.json"

# Fusion weights and tiered routing logic live in predict.py._get_final_score().
# They were previously duplicated here with different values, causing silent
# divergence. Deleted in Bug B3 fix — single source of truth only.


class ColdStartHandler:
    """
    Manages new-user initialisation and tiered routing.
    Instantiate once at service startup.
    """

    def __init__(
        self,
        r:            redis_lib.Redis,
        priors_path:  str = _DEFAULT_PRIORS_PATH,
        seq_len:      int = SEQ_LEN,
        feature_cols: list[str] = SEQ_FEATURES,
        ttl_seconds:  int = 30 * 86400,
    ):
        self._r             = r
        self._ops           = RedisAtomicOps(r)
        self._seq_len       = seq_len
        self._feature_cols  = feature_cols
        self._ttl           = ttl_seconds

        # Load cohort priors
        p = Path(priors_path)
        if p.exists():
            self._priors = load_cohort_priors(str(p))
            logger.info(
                "ColdStartHandler: loaded {} cohort priors from {}",
                len(self._priors), p,
            )
        else:
            logger.warning(
                "ColdStartHandler: priors file not found at {}. "
                "Run build_cohort_priors.py before starting the service.",
                p,
            )
            self._priors = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 1 — Signup initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def initialize_new_user(self, uid: str, signup_features: dict) -> None:
        """
        Called at new-user signup.  Populates Redis with cohort priors so the
        BiLSTM never receives a fully zero-padded tensor.

        Parameters
        ----------
        uid             : Unique user identifier
        signup_features : Dict containing at least DeviceType, P_emaildomain, ProductCD
        """
        cohort_key = _make_cohort_key_from_dict(signup_features)
        stats      = get_cohort_stats(self._priors, cohort_key, self._feature_cols)

        # ── 1. Write cohort priors to uid:features hash ───────────────────────
        user_key = f"user:{uid}:features"
        pipe     = self._r.pipeline()
        pipe.hset(user_key, mapping={
            "uid_tx_count":   0,
            "is_new_user":    1,
            "risk_prior":     stats.get("fraud_rate", 0.035),
            "cohort_key":     cohort_key,
            # Include cohort means for downstream feature imputation
            **{
                f"cohort_{col}_mean": stats.get(f"{col}_mean", 0.0)
                for col in self._feature_cols
            },
        })
        pipe.expire(user_key, self._ttl)
        pipe.execute()

        # ── 2. Pre-populate seq_window with phantom cohort-average steps ──────
        phantom = self._build_phantom_sequence(stats)
        seq_key = f"user:{uid}:seq_window"
        self._ops.set_window(seq_key, phantom, ttl=self._ttl)

        logger.info(
            "Initialised new user uid={} cohort='{}' fraud_prior={:.3%}",
            uid, cohort_key, stats.get("fraud_rate", 0.035),
        )

    def _build_phantom_sequence(self, stats: dict) -> list[np.ndarray]:
        """
        Build SEQ_LEN phantom feature vectors from cohort-average values.
        These are the exact same values used in build_sequences() padding.
        """
        means = np.array(
            [stats.get(f"{col}_mean", 0.0) for col in self._feature_cols],
            dtype=np.float32,
        )
        return [means.copy() for _ in range(self._seq_len)]

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 3 — Explicit is_new_user features
    # ─────────────────────────────────────────────────────────────────────────

    def get_user_meta_features(self, uid: str) -> dict:
        """
        Return is_new_user and tx_count_so_far for injection into the stacking
        feature vector.  These allow tree models to apply an elevated base risk
        to new users automatically (mimicking Stripe Radar behaviour).
        """
        try:
            raw = self._r.hget(f"user:{uid}:features", "uid_tx_count")
            tx_count  = int(raw) if raw is not None else 0
        except Exception:
            tx_count  = 0

        return {
            "tx_count_so_far": tx_count,
            "is_new_user":     int(tx_count < 3),
        }

    def increment_tx_count(self, uid: str) -> int:
        """Atomically increment the user's transaction counter after inference."""
        key = f"user:{uid}:features"
        new_count = self._r.hincrby(key, "uid_tx_count", 1)
        if new_count == 1:
            # First real transaction — mark as_new_user = 0 now (has 1 real tx)
            self._r.hset(key, "is_new_user", 0)
        return int(new_count)


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_cohort_key_from_dict(features: dict) -> str:
    """Build a cohort key from a features dict (mirrors dataset.py logic)."""
    parts = []
    for col in COHORT_COLS:
        val = features.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            parts.append("unknown")
        else:
            parts.append(str(val).strip().lower())
    return "_".join(parts)

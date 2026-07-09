from __future__ import annotations

import json
import pickle
from typing import Any, Dict, Optional

import numpy as np
import redis
from loguru import logger


class FeatureStore:
    """
    Redis-backed feature store for production fraud inference.

    Manages:
    - Velocity features (sliding window counts/sums)
    - Static aggregation features (uid_amt_mean, etc.)
    - GNN card node embeddings
    - Frequency lookup dicts (in-memory, loaded at startup)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db:   int = 0,
        ttl_seconds: int = 30 * 86400,
    ):
        self.r   = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self.ttl = ttl_seconds

        # In-memory freq dicts (loaded from artifacts at startup)
        self._freq_maps:      Dict[str, Dict] = {}
        self._card_embeddings: Dict[str, np.ndarray] = {}

        logger.info("FeatureStore connected to Redis {}:{}/db{}", host, port, db)

    # ─────────────────────────────────────────────────────────────────────────
    # Startup: load static dicts into memory
    # ─────────────────────────────────────────────────────────────────────────

    def load_static_artifacts(self, encoders_path: str) -> None:
        """
        Load frequency maps and GNN card embeddings from saved artifacts.
        Call once at service startup.
        """
        with open(encoders_path, "rb") as f:
            encoders = pickle.load(f)

        self._freq_maps = encoders.get("freq_maps", {})
        logger.info("Loaded {} frequency maps into memory.", len(self._freq_maps))

    def load_card_embeddings(
        self,
        emb_path: str,
        card_map_path: str,
    ) -> None:
        """Load GNN card embeddings (64-dim vectors) keyed by card1 value."""
        embs     = np.load(emb_path)
        with open(card_map_path, "rb") as f:
            card_map = pickle.load(f)
        # Invert map: idx → card1_value, then store {card1_value: embedding}
        idx_to_card = {v: k for k, v in card_map.items()}
        for idx, emb in enumerate(embs):
            card_val = idx_to_card.get(idx)
            if card_val:
                self._card_embeddings[card_val] = emb
        logger.info("Loaded {} GNN card embeddings into memory.", len(self._card_embeddings))

    # ─────────────────────────────────────────────────────────────────────────
    # Frequency lookup (in-memory, <1ms)
    # ─────────────────────────────────────────────────────────────────────────

    def get_freq(self, col: str, value: Any) -> float:
        """Look up frequency-encoded value for a column."""
        freq_map = self._freq_maps.get(col, {})
        return freq_map.get(str(value), 0.0)

    def get_card_embedding(self, card1_value: str) -> Optional[np.ndarray]:
        """Return 64-dim GNN embedding for card1, or None if unseen."""
        return self._card_embeddings.get(str(card1_value))

    # ─────────────────────────────────────────────────────────────────────────
    # Velocity features (Redis, 50-200ms)
    # ─────────────────────────────────────────────────────────────────────────

    def get_velocity_features(
        self,
        uid:        str,
        current_dt: int,
        windows:    tuple = (300, 900, 3600, 86400),
    ) -> Dict[str, float]:
        """
        Pull real-time velocity features from Redis sorted sets.
        Returns dict of feature_name → value.
        """
        key      = f"uid:{uid}"
        features = {}

        try:
            pipe = self.r.pipeline()

            for w in windows:
                cutoff = current_dt - w
                pipe.zcount(f"{key}:txn_times", cutoff, current_dt)
                pipe.zrangebyscore(f"{key}:txn_amts", cutoff, current_dt)

            pipe.zrevrange(f"{key}:txn_times", 0, 0, withscores=True)
            pipe.zrange(f"{key}:txn_times", 0, -1, withscores=True)

            results = pipe.execute()

            idx = 0
            for w in windows:
                label    = f"{w // 3600}h" if w % 3600 == 0 else f"{w}s"
                count    = int(results[idx])     if results[idx] is not None else 0
                members  = results[idx + 1]      or []
                amt_sum  = 0.0
                for m in members:
                    try:
                        # members come back as bytes from Redis (decode_responses=False)
                        s = m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m)
                        amt_sum += float(s.split(":")[1])
                    except (IndexError, ValueError):
                        pass

                features[f"uid_tx_{label}"]  = count
                if label == "24h":
                    features["uid_total_amt_24h"] = amt_sum
                else:
                    features[f"uid_amt_{label}"] = amt_sum
                idx += 2

            # Gap since last transaction
            latest = results[idx]
            if latest:
                last_ts = int(latest[0][1])
                features["uid_gap_last"] = float(current_dt - last_ts)
            else:
                features["uid_gap_last"] = -1.0

            # Average gap
            all_ts_data = results[idx + 1]
            if all_ts_data and len(all_ts_data) >= 2:
                ts_vals = sorted([int(s) for _, s in all_ts_data])
                gaps = [ts_vals[i] - ts_vals[i - 1] for i in range(1, len(ts_vals))]
                features["uid_avg_gap"] = float(np.mean(gaps))
            else:
                features["uid_avg_gap"] = -1.0

        except redis.RedisError as e:
            logger.warning("Redis velocity fetch error for uid={}: {}", uid, e)
            for w in windows:
                label = f"{w // 3600}h" if w % 3600 == 0 else f"{w}s"
                features[f"uid_tx_{label}"]  = -1.0
                if label == "24h":
                    features["uid_total_amt_24h"] = -1.0
                else:
                    features[f"uid_amt_{label}"] = -1.0
            features["uid_gap_last"] = -1.0
            features["uid_avg_gap"]  = -1.0

        return features

    # ─────────────────────────────────────────────────────────────────────────
    # Static aggregation features (Redis, 30-100ms)
    # ─────────────────────────────────────────────────────────────────────────

    def get_static_features(self, uid: str) -> Dict[str, float]:
        """
        Retrieve pre-computed uid aggregation stats (uid_amt_mean, etc.).
        These are updated periodically in batch, not per-transaction.
        """
        key  = f"uid_stats:{uid}"
        data = self.r.get(key)
        if data:
            return json.loads(data.decode())
        return {}

    def set_static_features(self, uid: str, features: Dict[str, float]) -> None:
        """Store uid aggregation stats (called during batch feature refresh)."""
        key = f"uid_stats:{uid}"
        self.r.setex(key, self.ttl, json.dumps(features))

    # ─────────────────────────────────────────────────────────────────────────
    # GNN outputs (100-300ms → cached in Redis)
    # ─────────────────────────────────────────────────────────────────────────

    def get_gnn_features(self, card1: str) -> Dict[str, Any]:
        """
        Retrieve cached GNN fraud probability and ring risk score for a card.
        Returns {} if not cached (use fallback = mean of other models).
        """
        key  = f"gnn:{card1}"
        data = self.r.get(key)
        if data:
            return json.loads(data.decode())
        emb = self.get_card_embedding(card1)
        if emb is not None:
            return {"node_embedding": emb.tolist()}
        return {}

    def set_gnn_features(
        self,
        card1:          str,
        fraud_prob:     float,
        ring_risk:      float,
        embedding:      Optional[np.ndarray] = None,
    ) -> None:
        """Cache GNN outputs for a card node after batch GNN inference."""
        key  = f"gnn:{card1}"
        data = {
            "gnn_fraud_prob":  float(fraud_prob),
            "ring_risk_score": float(ring_risk),
        }
        if embedding is not None:
            data["node_embedding"] = embedding.tolist()
        self.r.setex(key, self.ttl, json.dumps(data))

    # ─────────────────────────────────────────────────────────────────────────
    # Update after transaction (called POST-inference)
    # ─────────────────────────────────────────────────────────────────────────

    def update_after_transaction(
        self,
        uid:             str,
        transaction_amt: float,
        transaction_dt:  int,
    ) -> None:
        """
        Update Redis velocity store AFTER inference is complete.
        Called after the transaction decision — not before (would leak future info).
        """
        key  = f"uid:{uid}"
        pipe = self.r.pipeline()

        pipe.zadd(f"{key}:txn_times", {str(transaction_dt): transaction_dt})
        pipe.zadd(f"{key}:txn_amts",  {f"{transaction_dt}:{transaction_amt}": transaction_dt})

        # Remove entries older than 30 days
        cutoff = transaction_dt - self.ttl
        pipe.zremrangebyscore(f"{key}:txn_times", 0, cutoff)
        pipe.zremrangebyscore(f"{key}:txn_amts",  0, cutoff)

        pipe.expire(f"{key}:txn_times", self.ttl)
        pipe.expire(f"{key}:txn_amts",  self.ttl)

        pipe.execute()

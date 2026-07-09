from __future__ import annotations

import concurrent.futures
import json
import time
from typing import Dict, Optional

import numpy as np
import redis as redis_lib
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Module-level executor (one per process — created at import time)
# ─────────────────────────────────────────────────────────────────────────────

_executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="post-inference",
)

# Module-level shared state (set by init_post_inference)
_redis_client:   Optional[redis_lib.Redis] = None
_kafka_producer                            = None   # kafka.KafkaProducer | None
_kafka_topic:    str                       = "graph-updates"
_redis_atomic                              = None   # RedisAtomicOps | None
_feature_store                             = None   # FeatureStore | None


def init_post_inference(
    redis_client:             redis_lib.Redis,
    kafka_bootstrap_servers:  str  = "localhost:9092",
    kafka_topic:              str  = "graph-updates",
    kafka_enabled:            bool = True,
) -> None:
    """
    Initialise module-level shared resources.
    Call ONCE at service startup before any post_inference_async() calls.

    Parameters
    ----------
    redis_client            : Connected Redis client (same instance as FeatureStore)
    kafka_bootstrap_servers : Comma-separated Kafka broker list
    kafka_topic             : Topic for transaction events consumed by Neo4j sink
    kafka_enabled           : Set False to disable Kafka (e.g. local dev without Kafka)
    """
    global _redis_client, _kafka_producer, _kafka_topic
    global _redis_atomic, _feature_store

    _redis_client = redis_client
    _kafka_topic  = kafka_topic

    # ── Lazy-import to avoid hard dependency ─────────────────────────────────
    from src.inference.redis_atomic import RedisAtomicOps
    from src.inference.feature_store import FeatureStore

    _redis_atomic  = RedisAtomicOps(redis_client)
    _feature_store = FeatureStore.__new__(FeatureStore)
    # Manually set all attrs that __init__ would set — __new__ skips it.
    _feature_store.r                  = redis_client
    _feature_store.ttl                = 30 * 86400
    _feature_store._freq_maps         = {}   # not needed for velocity update
    _feature_store._card_embeddings   = {}   # not needed for velocity update


    if kafka_enabled:
        try:
            import warnings
            from kafka import KafkaProducer  # type: ignore
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                _kafka_producer = KafkaProducer(
                    bootstrap_servers=kafka_bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    acks="all",                  # wait for leader + in-sync replicas
                    retries=3,
                    linger_ms=5,                 # micro-batch for throughput
                    compression_type="gzip",
                )
            logger.info(
                "Kafka producer initialised (brokers={}, topic={}).",
                kafka_bootstrap_servers, kafka_topic,
            )
        except ImportError:
            logger.warning(
                "kafka-python not installed — Kafka task will be skipped. "
                "Install with: pip install kafka-python"
            )
            _kafka_producer = None
        except Exception as exc:
            logger.error(
                "Kafka producer init failed ({}). Kafka task disabled.", exc
            )
            _kafka_producer = None
    else:
        logger.info("Kafka disabled by config.")

    logger.info("post_inference initialised (executor max_workers=8).")


# ─────────────────────────────────────────────────────────────────────────────
# Task A — Update Redis velocity counters
# ─────────────────────────────────────────────────────────────────────────────

def update_velocity_counters(
    uid:             str,
    transaction_dt:  int,
    transaction_amt: float,
) -> None:
    """
    Add the completed transaction to Redis velocity sorted sets.

    Keys updated:
        uid:{uid}:txn_times  (ZSET — score = timestamp)
        uid:{uid}:txn_amts   (ZSET — member = "dt:amt", score = timestamp)

    Called AFTER the fraud score is returned (no future leakage).
    """
    if _feature_store is None:
        logger.warning("Task A: feature_store not initialised — call init_post_inference().")
        return
    try:
        _feature_store.update_after_transaction(uid, transaction_amt, transaction_dt)
        logger.debug("Task A: velocity counters updated for uid={}.", uid)
    except Exception as exc:
        logger.error("Task A: velocity update failed for uid={} — {}", uid, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Task B — Stream transaction to Neo4j via Kafka
# ─────────────────────────────────────────────────────────────────────────────

def stream_to_kafka_graph_topic(
    transaction: Dict,
    fraud_score: float,
) -> None:
    """
    Publish a transaction event to the Kafka topic consumed by the Neo4j
    Kafka connector.  The connector creates/updates (:Card)-[:USED_ON]->(:Device)
    relationships in the graph database.

    Message schema (JSON):
    {
        "card1":               <str>,
        "DeviceInfo":          <str>,
        "P_emaildomain":       <str>,
        "addr1":               <str | null>,
        "TransactionAmt":      <float>,
        "TransactionDT":       <int>,
        "ProductCD":           <str | null>,
        "hour_sin":            <float>,
        "hour_cos":            <float>,
        "uid_gap_last":        <float>,
        "final_fraud_score":   <float>,
        "timestamp":           <int>   ← wall-clock epoch for Neo4j TTL
    }
    """
    if _kafka_producer is None:
        logger.debug("Task B: Kafka producer not available — skipping.")
        return

    message = {
        "card1":             str(transaction.get("card1",         "unknown")),
        "DeviceInfo":        str(transaction.get("DeviceInfo",    "unknown")),
        "P_emaildomain":     str(transaction.get("P_emaildomain", "unknown")),
        "addr1":             str(transaction.get("addr1",         "unknown")),
        "TransactionAmt":    float(transaction.get("TransactionAmt", 0.0)),
        "TransactionDT":     int(transaction.get("TransactionDT",    0)),
        "ProductCD":         str(transaction.get("ProductCD",     "W")),
        "hour_sin":          float(transaction.get("hour_sin",    0.0)),
        "hour_cos":          float(transaction.get("hour_cos",    1.0)),
        "uid_gap_last":      float(transaction.get("uid_gap_last", 86400.0)),
        "final_fraud_score": float(fraud_score),
        "timestamp":         int(time.time()),
    }

    try:
        future = _kafka_producer.send(_kafka_topic, value=message)
        # Non-blocking: future resolves in background.
        # Log errors via callback instead of .get() which would block.
        future.add_errback(
            lambda exc: logger.error(
                "Task B: Kafka send failed for card={} — {}", message["card1"], exc
            )
        )
        logger.debug(
            "Task B: transaction streamed to Kafka topic={} card={}.",
            _kafka_topic, message["card1"],
        )
    except Exception as exc:
        logger.error("Task B: Kafka send error — {}", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Task C — Update ring signals (fast Redis-only detection)
# ─────────────────────────────────────────────────────────────────────────────

def update_ring_signals(
    uid:          str,
    device_id:    str,
    email_domain: str,
    transaction_dt: int,
    window_secs:  int = 300,
) -> Dict:
    """
    Run the fast Redis-based ring detector and persist the result back to
    the user's feature hash so the NEXT transaction can see it.

    This is the production wiring for RedisAtomicOps.fast_ring_check()
    which is defined but not called in predict_single().  Running it here
    post-inference means:
      - Zero added latency to the checkout path
      - Ring signal is fresh for the NEXT transaction from the same device

    Returns the ring result dict (for logging / monitoring), or {} on error.
    """
    if _redis_atomic is None or _redis_client is None:
        logger.warning("Task C: redis_atomic not initialised.")
        return {}

    try:
        result = _redis_atomic.fast_ring_check(
            uid=uid,
            device_id=device_id,
            email_domain=email_domain,
            window_secs=window_secs,
        )

        # Persist ring_signal into the user's feature hash so it's visible
        # to the next call to ColdStartHandler.get_user_meta_features()
        if result.get("ring_signal"):
            _redis_client.hset(
                f"user:{uid}:features",
                mapping={
                    "ring_signal":          result["ring_signal"],
                    "shared_device_users":  result["shared_device_users"],
                    "shared_email_users":   result["shared_email_users"],
                    "ring_checked_at":      transaction_dt,
                },
            )
            logger.warning(
                "Task C: ring signal set for uid={} "
                "(device_users={}, email_users={}).",
                uid,
                result["shared_device_users"],
                result["shared_email_users"],
            )
        else:
            logger.debug("Task C: no ring signal for uid={}.", uid)

        return result

    except Exception as exc:
        logger.error("Task C: ring check failed for uid={} — {}", uid, exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Task D — Update BiLSTM sequence window
# ─────────────────────────────────────────────────────────────────────────────

def update_seq_window(
    uid:            str,
    transaction:    Dict,
    stacking_score: float,
    seq_len:        int = 10,
) -> None:
    """
    Push the current transaction's feature vector into the user's BiLSTM
    sequence window in Redis.

    This is the safety-net counterpart to the atomic push inside predict_single().
    It is a no-op if predict_single() already pushed (the sequence is idempotent
    since each step is appended by TransactionDT order).

    Designed for callers that use a stateless/batch API and bypass predict_single()
    (e.g., bulk scoring pipelines).  In normal single-transaction flow, the push
    inside _predict_bilstm_from_redis() is sufficient and this task silently
    pushes a duplicate that is trimmed by LTRIM.

    Parameters
    ----------
    uid            : User identifier
    transaction    : Raw transaction dict (same schema as predict_single input)
    stacking_score : Stacking ensemble score for this transaction
    seq_len        : Must match SEQ_LEN used in BiLSTM training (default 10)
    """
    if _redis_atomic is None:
        logger.warning("Task D: redis_atomic not initialised.")
        return

    try:
        from src.models.bilstm.dataset import SEQ_FEATURES

        current_vec = np.zeros(len(SEQ_FEATURES), dtype=np.float32)
        for j, col in enumerate(SEQ_FEATURES):
            if col == "calibrated_stacking_score":
                current_vec[j] = stacking_score
            elif col in transaction:
                try:
                    current_vec[j] = float(transaction[col])
                except (ValueError, TypeError):
                    current_vec[j] = 0.0

        seq_key = f"user:{uid}:seq_window"
        _redis_atomic.push_and_get_window(seq_key, current_vec, max_len=seq_len)
        logger.debug("Task D: seq window updated for uid={}.", uid)

    except Exception as exc:
        logger.error("Task D: seq window update failed for uid={} — {}", uid, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Task E — Increment User Transaction Count
# ─────────────────────────────────────────────────────────────────────────────

def update_tx_count(uid: str) -> None:
    """
    Increment the user's transaction counter in Redis.
    This allows predict.py to correctly route returning users to the BiLSTM.
    """
    if _redis_client is None:
        return
    try:
        key = f"user:{uid}:features"
        new_count = _redis_client.hincrby(key, "uid_tx_count", 1)
        if new_count == 1:
            _redis_client.hset(key, "is_new_user", 0)
        logger.debug("Task E: tx count updated for uid={} (new_count={}).", uid, new_count)
    except Exception as exc:
        logger.error("Task E: tx count update failed for uid={} — {}", uid, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — post_inference_async
# ─────────────────────────────────────────────────────────────────────────────

def post_inference_async(
    uid:            str,
    transaction:    Dict,
    fraud_score:    float,
    stacking_score: float = 0.5,
) -> None:
    """
    Submit all four post-inference tasks to the background executor.

    The checkout API calls this function and returns the fraud_score
    to the user IMMEDIATELY.  This function itself returns in <1ms.
    All four tasks run in background threads.

    Parameters
    ----------
    uid            : User identifier (uid1 from transaction)
    transaction    : Raw transaction dict passed to predict_single()
    fraud_score    : Final fraud probability returned by predict_single()
    stacking_score : Intermediate stacking score (for seq window feature)
    """
    dt  = int(transaction.get("TransactionDT", int(time.time())))
    amt = float(transaction.get("TransactionAmt", 0.0))

    # Task A — Redis velocity counters
    _executor.submit(
        update_velocity_counters,
        uid,
        dt,
        amt,
    )

    # Task B — Kafka → Neo4j graph stream
    _executor.submit(
        stream_to_kafka_graph_topic,
        transaction,
        fraud_score,
    )

    # Task C — Redis ring signal detection
    _executor.submit(
        update_ring_signals,
        uid,
        str(transaction.get("DeviceInfo",    "unknown")),
        str(transaction.get("P_emaildomain", "unknown")),
        dt,
    )

    # Task D — BiLSTM sequence window (safety net)
    _executor.submit(
        update_seq_window,
        uid,
        transaction,
        stacking_score,
    )

    # Task E — Increment User Tx Count
    _executor.submit(
        update_tx_count,
        uid,
    )

    logger.debug(
        "post_inference_async: 4 tasks submitted for uid={} fraud_score={:.4f}.",
        uid, fraud_score,
    )


def shutdown(wait: bool = True) -> None:
    """
    Gracefully shut down the background executor.
    Call at service teardown (e.g. FastAPI lifespan on_shutdown).

    Parameters
    ----------
    wait : If True, block until all queued tasks finish.
    """
    logger.info("post_inference: shutting down executor (wait={})…", wait)
    _executor.shutdown(wait=wait)
    if _kafka_producer is not None:
        _kafka_producer.flush()
        _kafka_producer.close()
        logger.info("post_inference: Kafka producer closed.")

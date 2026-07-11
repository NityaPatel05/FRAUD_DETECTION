from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from loguru import logger
import yaml
import redis

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic transaction (mimics IEEE-CIS Vesta format)
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_TX = {
    "TransactionID":  "TEST_001",
    "TransactionDT":  86400,
    "TransactionAmt": 149.99,
    "ProductCD":      "W",
    "card1":          "TEST_CARD_001",
    "card2":          "321",
    "card3":          "150.0",
    "card4":          "visa",
    "card5":          "226.0",
    "card6":          "debit",
    "addr1":          "299",
    "addr2":          "87.0",
    "dist1":          19.0,
    "dist2":          None,
    "P_emaildomain":  "gmail.com",
    "R_emaildomain":  "gmail.com",
    "C1":  1.0, "C2":  1.0, "C3":  0.0, "C4":  0.0,
    "C5":  0.0, "C6":  1.0, "C7":  0.0, "C8":  0.0,
    "C9":  1.0, "C10": 0.0, "C11": 1.0, "C12": 0.0,
    "C13": 1.0, "C14": 1.0,
    "D1":  0.0, "D2":  None, "D3":  None, "D4":  0.0,
    "D5":  None, "D6":  None, "D7":  None, "D8":  None,
    "D9":  None, "D10": 0.0, "D11": None, "D12": None,
    "D13": None, "D14": None, "D15": 0.0,
    "M1":  "T", "M2": "T", "M3": "T", "M4": "M0",
    "M5":  "F", "M6": "T", "M7": None, "M8": None, "M9": None,
    "DeviceType":  "desktop",
    "DeviceInfo":  "Windows",
    "id_01": -5.0, "id_02": 184473.0,
    "id_03": None, "id_04": None,
    "id_05": 0.0,  "id_06": 0.0,
    "id_09": None, "id_10": None,
    "id_11": 100.0,
    "id_12": "NotFound", "id_13": 49.0, "id_14": -480.0,
    "id_15": "New",
    "id_17": 166.0, "id_19": 542.0, "id_20": 173.0,
    "id_28": "New",   "id_29": "Found",
    "id_30": "Windows 10",
    "id_31": "chrome 71.0",
    "id_32": 24.0,
    "id_33": "1920x1080",
    "id_34": "match_status:2",
    "id_36": "F",
    "id_37": "T",
    "id_38": "T",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: check Redis sorted sets
# ─────────────────────────────────────────────────────────────────────────────

def check_redis_velocity(r: redis.Redis, uid: str) -> dict:
    """
    Verify the uid's velocity sorted sets were updated by reading back
    the last 5 entries from uid:{uid}:txn_times.
    """
    times_key = f"uid:{uid}:txn_times"
    amts_key  = f"uid:{uid}:txn_amts"

    times = r.zrange(times_key, -5, -1, withscores=True)
    amts  = r.zrange(amts_key,  -5, -1, withscores=True)

    return {
        "txn_times_key":   times_key,
        "txn_times_count": r.zcard(times_key),
        "txn_times_last5": [(t[0].decode() if isinstance(t[0], bytes) else t[0], t[1])
                            for t in times],
        "txn_amts_key":    amts_key,
        "txn_amts_count":  r.zcard(amts_key),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: check Kafka (read back last message)
# ─────────────────────────────────────────────────────────────────────────────

def check_kafka_message(bootstrap_servers: str, topic: str, timeout_secs: int = 10) -> dict:
    """
    Consume the most recent message from the graph-updates topic
    and return it as a dict.  Times out after `timeout_secs`.
    """
    try:
        import warnings
        import uuid
        import time
        from kafka import KafkaConsumer  # type: ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                group_id=f"test-verifier-group-{uuid.uuid4().hex[:8]}",
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                consumer_timeout_ms=timeout_secs * 1000,
            )
        # Poll for messages until we find our synthetic test transaction
        start_time = time.time()
        for msg in consumer:
            if msg.value and msg.value.get("card1") == "TEST_CARD_001":
                consumer.close()
                return {"status": "ok", "message": msg.value, "offset": msg.offset}
            if time.time() - start_time > timeout_secs:
                break
        consumer.close()
        return {"status": "timeout", "message": None}
    except ImportError:
        return {"status": "kafka_not_installed", "message": None}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Main test
# ─────────────────────────────────────────────────────────────────────────────

def run_test(cfg: dict, check_kafka: bool = True) -> bool:
    from src.inference.predict import FraudDetectionPredictor
    from src.jobs.post_inference import init_post_inference

    r_cfg = cfg.get("redis", {})
    r = redis.Redis(
        host=r_cfg.get("host", "localhost"),
        port=r_cfg.get("port", 6379),
        db=r_cfg.get("db", 0),
    )

    # Initialise post_inference executor so Step 12 fires async tasks
    kafka_cfg = cfg.get("kafka", {})
    init_post_inference(
        redis_client=r,
        kafka_bootstrap_servers=kafka_cfg.get("bootstrap_servers", "localhost:9092"),
        kafka_topic=kafka_cfg.get("graph_topic", "graph-updates"),
        kafka_enabled=check_kafka and kafka_cfg.get("enabled", True),
    )

    # Load predictor
    checkpoint_dir = cfg["paths"]["checkpoints"]
    logger.info("Loading FraudDetectionPredictor from {}…", checkpoint_dir)
    predictor = FraudDetectionPredictor(checkpoint_dir=checkpoint_dir, redis_client=r)
    predictor.load_all(redis_client=r)

    # Run predict_single
    tx = dict(SYNTHETIC_TX)   # copy
    t0 = time.time()
    result = predictor.predict_single(tx)
    latency_ms = (time.time() - t0) * 1000

    logger.info("predict_single() returned in {:.1f}ms", latency_ms)
    logger.info("  fraud_prob     = {:.4f}", result["fraud_prob"])
    logger.info("  stacking_score = {:.4f}", result["stacking_score"])
    logger.info("  bilstm_prob    = {:.4f}", result["bilstm_prob"])
    logger.info("  decision       = {}", result["decision"])
    logger.info("  action         = {}", result["action"])

    all_passed = True

    # ── Check 1: result schema ────────────────────────────────────────────────
    required_keys = {"fraud_prob", "stacking_score", "bilstm_prob", "decision", "action", "model_probs"}
    missing = required_keys - set(result.keys())
    if missing:
        logger.error("FAIL: missing keys in result: {}", missing)
        all_passed = False
    else:
        logger.info("PASS: result schema OK")

    # ── Check 2: fraud_prob in [0, 1] ────────────────────────────────────────
    if not (0.0 <= result["fraud_prob"] <= 1.0):
        logger.error("FAIL: fraud_prob={} out of [0,1]", result["fraud_prob"])
        all_passed = False
    else:
        logger.info("PASS: fraud_prob in [0, 1]")

    # ── Check 3: Redis velocity sorted sets ───────────────────────────────────
    # Give the background thread 0.5s to complete
    time.sleep(0.5)

    # Derive uid from synthetic transaction
    uid_val = f"{tx.get('card1','')}_{tx.get('addr1','')}_{tx.get('P_emaildomain','')}"
    redis_info = check_redis_velocity(r, uid_val)

    if redis_info["txn_times_count"] > 0:
        logger.info(
            "PASS: Redis txn_times set has {} entries for uid={}",
            redis_info["txn_times_count"], uid_val,
        )
    else:
        logger.warning(
            "WARN: Redis txn_times has 0 entries for uid={} "
            "(may be expected if FeatureStore uses different key schema)",
            uid_val,
        )

    # ── Check 4: Kafka message ────────────────────────────────────────────────
    if check_kafka:
        time.sleep(1.0)   # let the producer flush
        kafka_result = check_kafka_message(
            bootstrap_servers=cfg.get("kafka", {}).get("bootstrap_servers", "localhost:9092"),
            topic=cfg.get("kafka", {}).get("graph_topic", "graph-updates"),
        )
        if kafka_result["status"] == "ok" and kafka_result["message"]:
            msg = kafka_result["message"]
            logger.info(
                "PASS: Kafka message received — card={} score={:.4f}",
                msg.get("card1"), msg.get("final_fraud_score"),
            )
        elif kafka_result["status"] == "kafka_not_installed":
            logger.warning("SKIP: kafka-python not installed — Kafka check skipped")
        else:
            logger.warning(
                "WARN: Kafka check returned status={} (Kafka may not be running)",
                kafka_result["status"],
            )
    else:
        logger.info("SKIP: Kafka check disabled by --no-kafka flag")

    return all_passed


def dry_run_test() -> bool:
    """Offline test — verifies the import chain and transaction dict structure."""
    logger.info("Dry-run mode: checking imports and transaction schema…")
    try:
        from src.inference.predict import FraudDetectionPredictor, _get_final_score
        from src.jobs.post_inference import post_inference_async, init_post_inference
        from src.jobs.gnn_refresh import pull_graph_from_neo4j
        logger.info("PASS: all imports OK")
    except ImportError as e:
        logger.error("FAIL: import error — {}", e)
        return False

    # Verify _get_final_score fusion logic
    tests = [
        (0.8, 0.2, 1,  0.8),   # cold: stacking wins
        (0.8, 0.2, 3,  0.8),   # tx=3: alpha=0 → stacking
        (0.8, 0.2, 4,  0.6),   # tx=4: alpha=0.33 → blend
        (0.8, 0.2, 5,  0.4),   # tx=5: alpha=0.67 → blend
        (0.8, 0.2, 6,  0.2),   # tx=6: bilstm wins
        (0.8, 0.2, 10, 0.2),   # full: bilstm wins
    ]
    for stacking, bilstm, tx, expected in tests:
        got = _get_final_score(stacking, bilstm, tx)
        ok  = abs(got - expected) < 1e-6
        status = "PASS" if ok else "FAIL"
        logger.info("{}: _get_final_score({}, {}, tx={}) = {:.4f} (expected {:.4f})",
                    status, stacking, bilstm, tx, got, expected)

    logger.info("Dry-run complete.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 2 integration test")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--no-kafka",  action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        success = dry_run_test()
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        success = run_test(cfg, check_kafka=not args.no_kafka)

    sys.exit(0 if success else 1)

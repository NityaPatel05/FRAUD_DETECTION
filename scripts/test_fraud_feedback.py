from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import Thread

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _create_test_db(db_url: str):
    from src.db.schema import create_all_tables, get_engine
    engine = get_engine(db_url)
    create_all_tables(engine)
    return engine


def _insert_test_transaction(engine, transaction_id: str) -> None:
    from src.db.schema import SessionLocal, TrainingTransaction
    session = SessionLocal(engine)
    try:
        row = TrainingTransaction(
            transaction_id   = transaction_id,
            uid              = "test_uid_001",
            transaction_dt   = 86400,
            transaction_amt  = 149.99,
            card1            = "TEST_CARD_001",
            device_info      = "Windows",
            p_emaildomain    = "gmail.com",
            final_fraud_score = 0.73,
            decision         = "flag_review",
            stacking_score   = 0.65,
            bilstm_prob      = 0.80,
            confirmed_fraud  = None,   # unlabelled — will be filled by consumer
        )
        session.add(row)
        session.commit()
        logger.info("Inserted test transaction: {}", transaction_id)
    finally:
        session.close()


def _read_confirmed_label(engine, transaction_id: str) -> dict:
    from src.db.schema import SessionLocal, TrainingTransaction
    session = SessionLocal(engine)
    try:
        row = session.query(TrainingTransaction).filter_by(
            transaction_id=transaction_id
        ).one_or_none()
        if row is None:
            return {"found": False}
        return {
            "found":          True,
            "confirmed_fraud": row.confirmed_fraud,
            "label_source":   row.label_source,
            "labelled_at":    row.labelled_at,
        }
    finally:
        session.close()


def _publish_kafka_label(
    bootstrap_servers: str,
    topic:             str,
    transaction_id:    str,
    is_fraud:          int = 1,
    label_source:      str = "chargeback",
) -> bool:
    try:
        import warnings
        from kafka import KafkaProducer  # type: ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        msg = {
            "transaction_id": transaction_id,
            "is_fraud":       is_fraud,
            "label_source":   label_source,
            "labelled_at":    datetime.utcnow().isoformat(),
            "notes":          "Automated test chargeback",
        }
        future = producer.send(topic, value=msg)
        future.get(timeout=10)   # block until sent
        producer.close()
        logger.info(
            "Published to Kafka topic={} txn={} is_fraud={}",
            topic, transaction_id, is_fraud,
        )
        return True
    except ImportError:
        logger.warning("kafka-python not installed — Kafka publish skipped")
        return False
    except Exception as exc:
        logger.error("Kafka publish failed: {}", exc)
        return False


def _run_consumer_briefly(db_url: str, bootstrap_servers: str, topic: str, secs: int = 8) -> None:
    """Run the fraud_label_consumer in a background thread for `secs` seconds, then stop."""
    from src.jobs.fraud_label_consumer import FraudLabelConsumer
    consumer = FraudLabelConsumer(
        database_url=db_url,
        kafka_bootstrap_servers=bootstrap_servers,
        kafka_topic=topic,
        kafka_group_id=f"test-consumer-{uuid.uuid4().hex[:8]}",
    )

    def _stopper():
        time.sleep(secs)
        consumer._running = False

    Thread(target=_stopper, daemon=True).start()
    consumer.run()   # blocks until _running=False


# ─────────────────────────────────────────────────────────────────────────────
# Test: DB-only (no Kafka)
# ─────────────────────────────────────────────────────────────────────────────

def test_db_only(db_url: str) -> bool:
    """
    Directly call apply_fraud_label() and verify the DB write.
    Tests the DB layer without touching Kafka.
    """
    from src.db.schema import apply_fraud_label, SessionLocal

    txn_id = f"TEST_{uuid.uuid4().hex[:8]}"
    engine = _create_test_db(db_url)
    _insert_test_transaction(engine, txn_id)

    # Apply label directly (simulates what fraud_label_consumer.py does)
    session = SessionLocal(engine)
    try:
        row = apply_fraud_label(
            session=session,
            transaction_id=txn_id,
            is_fraud=1,
            label_source="manual_test",
            labelled_at=datetime.utcnow(),
            notes="Direct DB test",
        )
        session.commit()
        logger.info(
            "apply_fraud_label() returned: {} confirmed_fraud={}",
            row, row.confirmed_fraud if row else None,
        )
    finally:
        session.close()

    # Verify
    result = _read_confirmed_label(engine, txn_id)
    if result["found"] and result["confirmed_fraud"] == 1:
        logger.info("PASS: confirmed_fraud=1 written to DB for txn={}", txn_id)
        return True
    else:
        logger.error("FAIL: DB read-back unexpected: {}", result)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Test: Full Kafka → consumer → DB round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_full_roundtrip(
    db_url:            str,
    bootstrap_servers: str,
    label_topic:       str,
) -> bool:
    txn_id = f"TEST_{uuid.uuid4().hex[:8]}"
    logger.info("Round-trip test using txn_id={}", txn_id)

    # 1. Insert unlabelled transaction
    engine = _create_test_db(db_url)
    _insert_test_transaction(engine, txn_id)

    before = _read_confirmed_label(engine, txn_id)
    logger.info("Before publish: confirmed_fraud={}", before.get("confirmed_fraud"))

    # 2. Publish Kafka message
    kafka_ok = _publish_kafka_label(
        bootstrap_servers=bootstrap_servers,
        topic=label_topic,
        transaction_id=txn_id,
        is_fraud=1,
        label_source="chargeback",
    )
    if not kafka_ok:
        logger.warning("Kafka publish failed — falling back to DB-only test")
        return test_db_only(db_url)

    # 3. Run consumer for 8 seconds (long enough to pick up the message)
    logger.info("Starting fraud_label_consumer for 8s to consume the message…")
    _run_consumer_briefly(db_url, bootstrap_servers, label_topic, secs=8)

    # 4. Verify DB
    after = _read_confirmed_label(engine, txn_id)
    logger.info("After consumer: confirmed_fraud={} source={} labelled_at={}",
                after.get("confirmed_fraud"),
                after.get("label_source"),
                after.get("labelled_at"))

    if after["found"] and after["confirmed_fraud"] == 1:
        logger.info("PASS: confirmed_fraud=1 written for txn={}", txn_id)
        return True
    else:
        logger.error(
            "FAIL: Expected confirmed_fraud=1, got: {}", after.get("confirmed_fraud")
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 3 fraud feedback integration test")
    parser.add_argument("--config",       default="configs/config.yaml")
    parser.add_argument("--sqlite",       action="store_true",
                        help="Use SQLite instead of PostgreSQL")
    parser.add_argument("--db-only",      action="store_true",
                        help="Test DB layer only, no Kafka")
    parser.add_argument("--publish-only", action="store_true",
                        help="Only publish a Kafka message, don't run consumer")
    args = parser.parse_args()

    # ── Build DB URL ──────────────────────────────────────────────────────────
    if args.sqlite:
        db_url = "sqlite:///fraud_test.db"
        logger.info("Using SQLite: {}", db_url)
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        db_url = cfg["database"]["url"]

    # ── Route to correct test mode ────────────────────────────────────────────
    if args.db_only or args.sqlite:
        success = test_db_only(db_url)

    elif args.publish_only:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        txn_id = f"TEST_{uuid.uuid4().hex[:8]}"
        engine = _create_test_db(db_url)
        _insert_test_transaction(engine, txn_id)
        ok = _publish_kafka_label(
            bootstrap_servers=cfg["kafka"]["bootstrap_servers"],
            topic=cfg["kafka"]["label_topic"],
            transaction_id=txn_id,
        )
        logger.info("Message published (txn_id={}) — check DB manually with:", txn_id)
        logger.info("  SELECT * FROM training_transactions WHERE transaction_id='{}';", txn_id)
        success = ok

    else:
        # Full round-trip
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        success = test_full_roundtrip(
            db_url=db_url,
            bootstrap_servers=cfg["kafka"]["bootstrap_servers"],
            label_topic=cfg["kafka"]["label_topic"],
        )

    logger.info("=" * 50)
    logger.info("Result: {}", "PASSED" if success else "FAILED")
    sys.exit(0 if success else 1)

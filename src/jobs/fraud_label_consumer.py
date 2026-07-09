from __future__ import annotations

import json
import signal
import sys
from datetime import datetime
from typing import Optional

import yaml
from loguru import logger

from src.db.schema import (
    apply_fraud_label,
    create_all_tables,
    get_engine,
    SessionLocal,
)


# ─────────────────────────────────────────────────────────────────────────────
# Consumer
# ─────────────────────────────────────────────────────────────────────────────

class FraudLabelConsumer:
    """
    Kafka consumer that writes confirmed fraud labels to PostgreSQL.

    Parameters
    ----------
    database_url             : SQLAlchemy connection string
    kafka_bootstrap_servers  : Comma-separated broker list
    kafka_topic              : Topic to consume (default: "confirmed_fraud_labels")
    kafka_group_id           : Consumer group ID (unique per deployment)
    """

    def __init__(
        self,
        database_url:            str,
        kafka_bootstrap_servers: str = "localhost:9092",
        kafka_topic:             str = "confirmed_fraud_labels",
        kafka_group_id:          str = "fraud-label-consumer-v1",
        max_poll_records:        int = 100,
    ):
        self.database_url   = database_url
        self.kafka_servers  = kafka_bootstrap_servers
        self.kafka_topic    = kafka_topic
        self.kafka_group_id = kafka_group_id
        self.max_poll_records = max_poll_records

        self._running = True
        self._consumer = None
        self._engine   = None

        # Graceful shutdown on SIGTERM / SIGINT
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT,  self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        logger.info("FraudLabelConsumer: shutdown signal received, stopping…")
        self._running = False

    def _init_kafka(self):
        """Create Kafka consumer (lazy — called at run() time)."""
        try:
            from kafka import KafkaConsumer  # type: ignore
            self._consumer = KafkaConsumer(
                self.kafka_topic,
                bootstrap_servers=self.kafka_servers,
                group_id=self.kafka_group_id,
                auto_offset_reset="earliest",
                enable_auto_commit=False,   # manual commit after DB write
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                max_poll_records=self.max_poll_records,
                session_timeout_ms=30_000,
                heartbeat_interval_ms=10_000,
            )
            logger.info(
                "FraudLabelConsumer: Kafka consumer connected (topic={}, group={}).",
                self.kafka_topic, self.kafka_group_id,
            )
        except ImportError:
            logger.error(
                "kafka-python not installed. "
                "Run: pip install kafka-python"
            )
            sys.exit(1)
        except Exception as exc:
            logger.error("FraudLabelConsumer: Kafka init failed — {}", exc)
            raise

    def _init_db(self):
        """Create DB engine and ensure tables exist."""
        self._engine = get_engine(self.database_url)
        create_all_tables(self._engine)

    # ── Message processing ────────────────────────────────────────────────────

    def _parse_message(self, value: dict) -> Optional[dict]:
        """
        Validate and parse an incoming Kafka message.
        Returns None if the message is malformed (will be logged and skipped).
        """
        required = {"transaction_id", "is_fraud"}
        missing  = required - set(value.keys())
        if missing:
            logger.warning("FraudLabelConsumer: missing fields {} in message={}", missing, value)
            return None

        try:
            labelled_at = (
                datetime.fromisoformat(value["labelled_at"])
                if "labelled_at" in value and value["labelled_at"]
                else datetime.utcnow()
            )
        except ValueError:
            labelled_at = datetime.utcnow()

        return {
            "transaction_id": str(value["transaction_id"]),
            "is_fraud":       int(value["is_fraud"]),
            "label_source":   str(value.get("label_source", "kafka")),
            "labelled_at":    labelled_at,
            "notes":          value.get("notes"),
        }

    def _process_batch(self, records: dict) -> tuple[int, int]:
        """
        Write a batch of Kafka messages to the DB.
        Returns (n_written, n_skipped).
        """
        n_written = n_skipped = 0
        session = SessionLocal(self._engine)

        try:
            for tp, messages in records.items():
                for msg in messages:
                    parsed = self._parse_message(msg.value)
                    if parsed is None:
                        n_skipped += 1
                        continue

                    row = apply_fraud_label(session=session, **parsed)
                    if row is not None:
                        n_written += 1
                        logger.debug(
                            "Label written: txn={} is_fraud={} source={}",
                            parsed["transaction_id"],
                            parsed["is_fraud"],
                            parsed["label_source"],
                        )
                    else:
                        n_skipped += 1

            session.commit()

        except Exception as exc:
            session.rollback()
            logger.error("FraudLabelConsumer: DB write failed — {}. Rolling back.", exc)
            raise   # re-raise so Kafka offset is NOT committed
        finally:
            session.close()

        return n_written, n_skipped

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Blocking consume loop.  Stops cleanly on SIGTERM / SIGINT.
        """
        self._init_db()
        self._init_kafka()

        logger.info(
            "FraudLabelConsumer: starting consume loop on topic={}.", self.kafka_topic
        )
        total_written = total_skipped = 0

        while self._running:
            try:
                records = self._consumer.poll(timeout_ms=1000, max_records=self.max_poll_records)
                if not records:
                    continue

                n_written, n_skipped = self._process_batch(records)
                total_written += n_written
                total_skipped += n_skipped

                # Commit offsets only AFTER successful DB write
                self._consumer.commit()

                if n_written > 0:
                    logger.info(
                        "FraudLabelConsumer: batch done — written={} skipped={} total_written={}",
                        n_written, n_skipped, total_written,
                    )

            except Exception as exc:
                logger.error(
                    "FraudLabelConsumer: unhandled error — {}. Will retry on next poll.", exc
                )

        logger.info(
            "FraudLabelConsumer: stopped. total_written={} total_skipped={}",
            total_written, total_skipped,
        )
        self._consumer.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Confirmed fraud label Kafka consumer")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--topic",  default=None, help="Override Kafka topic from config")
    parser.add_argument("--group",  default=None, help="Override consumer group ID")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    consumer = FraudLabelConsumer(
        database_url=cfg["database"]["url"],
        kafka_bootstrap_servers=cfg["kafka"]["bootstrap_servers"],
        kafka_topic=args.topic  or cfg["kafka"].get("label_topic", "confirmed_fraud_labels"),
        kafka_group_id=args.group or "fraud-label-consumer-v1",
    )
    consumer.run()

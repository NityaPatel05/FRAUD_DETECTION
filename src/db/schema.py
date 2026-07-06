from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float,
    Integer, String, Text, create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Base + engine factory
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


def get_engine(database_url: str, **kwargs):
    """
    Create a SQLAlchemy engine.

    Parameters
    ----------
    database_url : e.g. "postgresql://fraud:pass@localhost:5432/fraud_db"
                   or   "sqlite:///fraud.db"  (for local testing)
    """
    engine = create_engine(
        database_url,
        pool_pre_ping=True,     # reconnect on stale connections
        pool_size=5,
        max_overflow=10,
        **kwargs,
    )
    logger.info("DB engine created for: {}", database_url.split("@")[-1])
    return engine


def create_all_tables(engine) -> None:
    """Create all tables that don't exist yet. Idempotent."""
    Base.metadata.create_all(engine)
    logger.info("All DB tables created / verified.")


def SessionLocal(engine) -> Session:
    """Return a new SQLAlchemy session bound to `engine`."""
    Factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return Factory()


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — training_transactions
# ─────────────────────────────────────────────────────────────────────────────

class TrainingTransaction(Base):
    """
    One row per scored transaction.

    Populated by two sources:
      1. predict_single() / predict_batch() — writes immediately after scoring.
      2. fraud_label_consumer.py           — fills confirmed_fraud when a label arrives.

    The weekly retrain job queries:
        WHERE confirmed_fraud IS NOT NULL
        AND   labelled_at > last_retrain_at
    """
    __tablename__ = "training_transactions"

    # ── Primary key ───────────────────────────────────────────────────────────
    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)

    # ── Transaction identity ──────────────────────────────────────────────────
    transaction_id  = Column(String(64),  nullable=False, index=True, unique=True)
    uid             = Column(String(128), nullable=False, index=True)
    transaction_dt  = Column(Integer,     nullable=False)   # seconds since epoch (kaggle format)
    transaction_amt = Column(Float,       nullable=False)

    # ── Entity identifiers ────────────────────────────────────────────────────
    card1         = Column(String(32),  nullable=True)
    card4         = Column(String(32),  nullable=True)   # Network (Visa/MC)
    card6         = Column(String(32),  nullable=True)   # Type (Credit/Debit)
    device_info   = Column(String(256), nullable=True)
    device_type   = Column(String(32),  nullable=True)   # Mobile/Desktop
    p_emaildomain = Column(String(128), nullable=True)
    r_emaildomain = Column(String(128), nullable=True)   # Receiver email
    addr1         = Column(String(32),  nullable=True)
    product_cd    = Column(String(4),   nullable=True)

    # ── Model output (written at score time) ──────────────────────────────────
    final_fraud_score = Column(Float,   nullable=False)
    stacking_score    = Column(Float,   nullable=True)
    bilstm_prob       = Column(Float,   nullable=True)
    decision          = Column(String(20), nullable=False)  # auto_approve / flag_review / block

    # ── Confirmed fraud label (filled by fraud_label_consumer.py) ─────────────
    # NULL  = unlabelled
    # 1     = confirmed fraud  (chargeback, manual review, etc.)
    # 0     = confirmed legitimate
    confirmed_fraud = Column(Integer,    nullable=True,  index=True)
    label_source    = Column(String(64), nullable=True)   # "chargeback", "manual_review", "auto"
    labelled_at     = Column(DateTime,   nullable=True,  index=True)
    label_notes     = Column(Text,       nullable=True)

    # ── Audit ─────────────────────────────────────────────────────────────────
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<TrainingTransaction id={self.id} "
            f"txn={self.transaction_id} "
            f"score={self.final_fraud_score:.3f} "
            f"label={self.confirmed_fraud}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 — model_versions
# ─────────────────────────────────────────────────────────────────────────────

class ModelVersion(Base):
    """
    Audit log of every trained model checkpoint.

    Lifecycle:
        shadow  → champion (promote after shadow AUC beats champion)
        champion → retired  (when a new champion is promoted)
    """
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # ── Version metadata ──────────────────────────────────────────────────────
    version_tag  = Column(String(64),  nullable=False, unique=True)   # e.g. "v20240701_001"
    model_type   = Column(String(32),  nullable=False)                # "xgb", "bilstm", "full_stack"
    checkpoint_path = Column(String(512), nullable=False)             # path to artifacts dir

    # ── Training provenance ───────────────────────────────────────────────────
    n_train_rows     = Column(Integer, nullable=True)
    n_new_labels     = Column(Integer, nullable=True)   # confirmed labels used in this run
    label_cutoff_dt  = Column(DateTime, nullable=True)  # last labelled_at included

    # ── Performance metrics ───────────────────────────────────────────────────
    val_auc_roc  = Column(Float, nullable=True)
    val_auc_pr   = Column(Float, nullable=True)
    shadow_auc_roc = Column(Float, nullable=True)  # AUC measured during shadow period

    # ── Deployment status ─────────────────────────────────────────────────────
    is_shadow   = Column(Boolean, default=False,  nullable=False)
    is_champion = Column(Boolean, default=False,  nullable=False)
    is_retired  = Column(Boolean, default=False,  nullable=False)
    promoted_at = Column(DateTime, nullable=True)
    retired_at  = Column(DateTime, nullable=True)

    # ── MLflow reference (optional) ───────────────────────────────────────────
    mlflow_run_id = Column(String(64), nullable=True)

    # ── Audit ─────────────────────────────────────────────────────────────────
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    trained_by = Column(String(64), nullable=True)   # "weekly_retrain_job", "manual"

    def __repr__(self) -> str:
        status = "champion" if self.is_champion else ("shadow" if self.is_shadow else "retired")
        return f"<ModelVersion {self.version_tag} [{status}] auc={self.val_auc_roc}>"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def upsert_transaction(
    session:     Session,
    transaction_id: str,
    uid:            str,
    transaction_dt: int,
    transaction_amt: float,
    final_fraud_score: float,
    decision:       str,
    stacking_score: Optional[float] = None,
    bilstm_prob:    Optional[float] = None,
    **extra_fields,
) -> TrainingTransaction:
    """
    Insert a new transaction row, or update if already present.
    Used by predict_single() / predict_batch() to record every score.
    """
    row = session.query(TrainingTransaction).filter_by(
        transaction_id=transaction_id
    ).one_or_none()

    if row is None:
        row = TrainingTransaction(
            transaction_id=transaction_id,
            uid=uid,
            transaction_dt=transaction_dt,
            transaction_amt=transaction_amt,
            final_fraud_score=final_fraud_score,
            decision=decision,
            stacking_score=stacking_score,
            bilstm_prob=bilstm_prob,
            **{k: v for k, v in extra_fields.items()
               if hasattr(TrainingTransaction, k)},
        )
        session.add(row)
    else:
        row.final_fraud_score = final_fraud_score
        row.decision          = decision
        row.stacking_score    = stacking_score
        row.bilstm_prob       = bilstm_prob
        row.updated_at        = datetime.utcnow()

    return row


def apply_fraud_label(
    session:        Session,
    transaction_id: str,
    is_fraud:       int,
    label_source:   str,
    labelled_at:    Optional[datetime] = None,
    notes:          Optional[str] = None,
) -> Optional[TrainingTransaction]:
    """
    Write a confirmed fraud label onto an existing transaction row.
    Called by fraud_label_consumer.py after receiving a Kafka message.
    Returns None if transaction_id is not found.
    """
    row = session.query(TrainingTransaction).filter_by(
        transaction_id=transaction_id
    ).one_or_none()

    if row is None:
        logger.warning(
            "apply_fraud_label: transaction_id={} not found in DB.", transaction_id
        )
        return None

    row.confirmed_fraud = int(is_fraud)
    row.label_source    = label_source
    row.labelled_at     = labelled_at or datetime.utcnow()
    row.label_notes     = notes
    row.updated_at      = datetime.utcnow()
    return row

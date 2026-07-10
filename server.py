from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict, Any

import redis
import yaml
from fastapi import FastAPI, HTTPException
from loguru import logger

from src.inference.predict import FraudDetectionPredictor
from src.jobs.post_inference import init_post_inference, shutdown


# ─────────────────────────────────────────────────────────────────────────────
# Global predictor (initialised once in lifespan)
# ─────────────────────────────────────────────────────────────────────────────
predictor: FraudDetectionPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    logger.info("Service starting up…")
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    redis_cfg = cfg.get("redis", {})
    r = redis.Redis(
        host=redis_cfg.get("host", "localhost"),
        port=redis_cfg.get("port", 6379),
        db=redis_cfg.get("db", 0),
        decode_responses=False,
    )

    # 1. Load all models — pass the shared Redis client so feature_store
    #    and the models use the same connection pool
    predictor = FraudDetectionPredictor(
        checkpoint_dir=cfg["paths"]["checkpoints"],
        redis_client=r,
    )
    predictor.load_all(redis_client=r)
    logger.info("All models loaded.")

    # 2. Initialise post-inference async executor + Kafka producer
    kafka_cfg = cfg.get("kafka", {})
    init_post_inference(
        redis_client=r,
        kafka_bootstrap_servers=kafka_cfg.get("bootstrap_servers", "localhost:9092"),
        kafka_topic=kafka_cfg.get("graph_topic", "graph-updates"),
        kafka_enabled=kafka_cfg.get("enabled", True),
    )
    logger.info("Post-inference executor ready.")

    yield   # ← service is live; FastAPI handles requests

    # Graceful shutdown: drain in-flight background tasks, flush Kafka
    logger.info("Service shutting down…")
    shutdown(wait=True)


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Liveness probe — returns OK if models are loaded."""
    return {
        "status": "ok",
        "models_loaded": predictor is not None,
    }


@app.post("/score")
def score(transaction: Dict[str, Any]) -> dict:
    """
    Score a single transaction for fraud probability.

    Expected fields (same schema as IEEE-CIS / Vesta training data):
        TransactionDT   : int   (seconds since epoch)
        TransactionAmt  : float
        card1           : str
        DeviceInfo      : str   (optional)
        P_emaildomain   : str   (optional)
        ProductCD       : str   (optional)
        ... (other feature fields are optional — missing ones default to NaN)

    Returns
    -------
    {
        "fraud_prob":      float,    # final calibrated fraud probability [0, 1]
        "stacking_score":  float,    # base meta-learner output
        "bilstm_prob":     float,    # BiLSTM sequential score
        "decision":        str,      # "auto_approve" | "flag_review" | "block"
        "action":          str,      # human-readable recommended action
        "model_probs": {
            "xgb":         float,
            "lgbm":        float,
            "catboost":    float,
            "stacking":    float,
            "bilstm":      float,
        },
        "latency_ms":      float,
    }
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    try:
        result = predictor.predict_single(dict(transaction))
        return result
    except Exception as exc:
        logger.error("Score endpoint error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/score/batch")
def score_batch(transactions: list[Dict[str, Any]]) -> list[dict]:
    """
    Score a list of transactions (vectorised — much faster than calling /score N times).
    Returns a list of result dicts in the same order as the input.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    if not transactions:
        return []

    import pandas as pd
    try:
        df = pd.DataFrame(transactions)
        results = predictor.predict_batch(df)
        return results
    except Exception as exc:
        logger.error("Batch score endpoint error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

"""
Phase 7 — v2 real-time scoring API.

  uvicorn src.serving.api_v2:app --port 8000

POST /v2/score   transaction JSON (raw IEEE-style fields) → decision (action, p, reasons, latency)
POST /v2/label   {"TransactionID": ..., "is_fraud": 0|1, "source": "chargeback|analyst|customer"}
GET  /v2/health  model version, rules loaded
POST /v2/snapshot  persist engine state (restart without replaying history)
The legacy server.py (old models) is left untouched.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.serving.service_factory import build_service

svc = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global svc
    svc = build_service()
    yield
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    svc.engine.save_state(cfg["decision_v2"]["state_snapshot"])


app = FastAPI(title="Fraud scoring v2", lifespan=lifespan)


class Label(BaseModel):
    TransactionID: int
    is_fraud: float
    source: str = "chargeback"


@app.post("/v2/score")
def score(txn: Dict[str, Any]):
    for k in ("TransactionID", "TransactionDT", "TransactionAmt"):
        if k not in txn:
            raise HTTPException(422, f"missing field {k}")
    return svc.decide(txn)


@app.post("/v2/label")
def label(lbl: Label):
    svc.record_label(lbl.TransactionID, lbl.is_fraud)
    return {"ok": True}


@app.get("/v2/health")
def health():
    return {"model_version": svc.bundle.version, "n_rules": len(svc.rules),
            "challenger": svc.challenger.version if svc.challenger else None}


@app.post("/v2/snapshot")
def snapshot():
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    svc.engine.save_state(cfg["decision_v2"]["state_snapshot"])
    return {"ok": True}

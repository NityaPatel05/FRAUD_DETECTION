"""
Phase 7 — v2 real-time scoring API.

  uvicorn src.serving.api_v2:app --port 8000

POST /v2/score     transaction JSON (gateway fields) -> decision (action, p, reasons, latency).
                   Idempotent per TransactionID: a repeat returns the cached decision with
                   idempotent_replay=true and changes no state (409 if the cache no longer has it).
POST /v2/label     fraud report {"TransactionID", "is_fraud": 1, "known_at"?: <TransactionDT units>}.
                   Only frauds need to be posted (chargeback feed): a transaction with no report becomes a
                   mature negative by itself after label_policy.maturity_days. Response carries a status:
                   registered | already_registered | negative_ignored | unknown_transaction | invalid
GET  /v2/health    model version, rules, state sizes, label counters
POST /v2/snapshot  persist engine state + fingerprint (restart without replaying history)

Sync endpoints run in FastAPI's threadpool; all shared state is behind the engine lock (see
online_features.py). Run ONE worker process per state shard: separate processes do not share state.
The legacy server.py (old models) is left untouched.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.serving.online_features import DuplicateTransactionError, MissingFieldsError
from src.serving.service_factory import build_service

svc = None


def _save(service) -> None:
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    service.engine.save_state(cfg["decision_v2"]["state_snapshot"], extra=getattr(service, "snapshot_extra", None))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global svc
    svc = build_service()
    yield
    _save(svc)


app = FastAPI(title="Fraud scoring v2", lifespan=lifespan)


class Label(BaseModel):
    TransactionID: int
    is_fraud: float
    source: str = "chargeback"
    known_at: Optional[int] = None          # real chargeback date (TransactionDT units); default = label policy


@app.post("/v2/score")
def score(txn: Dict[str, Any]):
    for k in ("TransactionID", "TransactionDT", "TransactionAmt"):
        if k not in txn:
            raise HTTPException(422, f"missing field {k}")
    try:
        return svc.decide(txn)
    except MissingFieldsError as e:
        raise HTTPException(422, str(e))
    except DuplicateTransactionError as e:
        raise HTTPException(409, str(e))


@app.post("/v2/label")
def label(lbl: Label):
    status = svc.record_label(lbl.TransactionID, lbl.is_fraud, lbl.known_at)
    return {"ok": status in ("registered", "already_registered", "negative_ignored"), "status": status}


@app.get("/v2/health")
def health():
    return {"model_version": svc.bundle.version, "n_rules": len(svc.rules),
            "challenger": svc.challenger.version if svc.challenger else None,
            "label_policy": svc.engine.policy.to_dict(), "label_stats": dict(svc.engine.label_stats),
            "state": svc.engine.state_size()}


@app.post("/v2/snapshot")
def snapshot():
    _save(svc)
    return {"ok": True}

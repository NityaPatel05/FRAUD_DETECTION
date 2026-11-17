"""Outcome accounting on a simulated transaction table. Ground truth (is_fraud_true) is used HERE ONLY, to score
what the decisions of a policy really cost; no policy ever sees it."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import COST_DEFAULTS

REVIEW_STATUSES = ("review_confirmed", "review_cleared")


def add_cost_columns(tx: pd.DataFrame, cfg_start: str, cost: Dict | None = None, explore_rate: float = 0.0) -> pd.DataFrame:
    """Per-row outcome costs (decided rows only). Returns a light frame with day, flags and USD columns."""
    c = dict(COST_DEFAULTS, **(cost or {}))
    d = tx[tx["txn_type"] != "refund"]
    amt = pd.to_numeric(d["amount_usd"], errors="coerce").astype("float64").fillna(0).to_numpy()
    fraud = d["is_fraud_true"].to_numpy() == 1
    app = d["approved"].to_numpy() == 1
    liab = d["liability_shift"].to_numpy() == 1
    by = d["declined_by"].to_numpy()
    dec = d["decision"].to_numpy()
    prop = pd.to_numeric(d["propensity"], errors="coerce").to_numpy()
    start = pd.Timestamp(cfg_start, tz="UTC")
    day = ((d["event_time"] - start).dt.total_seconds() // 86400).astype(int).to_numpy()
    loss_full = c["fraud_loss_frac"] * amt + c["chargeback_fee"]
    fraud_loss = np.where(app & fraud, loss_full * (1 - liab * c["liability_shift_share"]), 0.0)
    pol_decl = by == "policy"
    sca = by == "sca"
    rev_decl = by == "review"
    reviewed = np.isin(d["label_status"].to_numpy(), REVIEW_STATUSES)
    fd = np.where(~fraud & (pol_decl | rev_decl), c["margin_rate"] * amt + c["churn_cost"], 0.0)
    sca_cost = np.where(~fraud & sca, c["margin_rate"] * amt + c["stepup_churn_frac"] * c["churn_cost"], 0.0)
    stepups = (dec == "step_up_3ds") & (d["channel"].to_numpy() == "CNP")
    explored = ((dec == "approve") & np.isfinite(prop) & (prop < 1.0) & (explore_rate > 0)
                & np.isclose(prop, explore_rate, atol=1e-6))
    out = pd.DataFrame(dict(
        day=day, fraud=fraud, legit=~fraud, approved_fraud=app & fraud, blocked_fraud=~app & fraud,
        pol_false_decline=pol_decl & ~fraud, decl_any=pol_decl, review_load=reviewed, stepup=stepups,
        fraud_loss=fraud_loss, fd_cost=fd, sca_cost=sca_cost,
        review_cost=reviewed * c["review_cost"], stepup_cost=stepups * c["stepup_fee"],
        explored=explored, expl_fraud_loss=np.where(explored, fraud_loss, 0.0),
        expl_margin_gain=np.where(explored & ~fraud, c["margin_rate"] * amt, 0.0),
        scenario=d["fraud_scenario"].to_numpy()))
    out["net_cost"] = out[["fraud_loss", "fd_cost", "sca_cost", "review_cost", "stepup_cost"]].sum(axis=1)
    return out


def summarize(o: pd.DataFrame) -> Dict:
    n_legit, n_fraud = int(o["legit"].sum()), int(o["fraud"].sum())
    return dict(
        rows=int(len(o)), fraud_rows=n_fraud,
        fraud_loss_usd=float(o["fraud_loss"].sum()),
        fraud_block_rate=float(o["blocked_fraud"].sum() / max(n_fraud, 1)),
        false_decline_rate=float(o["pol_false_decline"].sum() / max(n_legit, 1)),
        false_declines=int(o["pol_false_decline"].sum()),
        false_decline_cost_usd=float(o["fd_cost"].sum()),
        sca_abandon_cost_usd=float(o["sca_cost"].sum()),
        review_cost_usd=float(o["review_cost"].sum()), reviews=int(o["review_load"].sum()),
        stepup_rate=float(o["stepup"].mean()), stepups=int(o["stepup"].sum()),
        review_rate=float(o["review_load"].mean()),
        exploration_rows=int(o["explored"].sum()),
        exploration_fraud_loss_usd=float(o["expl_fraud_loss"].sum()),
        exploration_net_cost_usd=float(o["expl_fraud_loss"].sum() - o["expl_margin_gain"].sum()),
        net_cost_usd=float(o["net_cost"].sum()),
    )


def by_period(o: pd.DataFrame, period_days: int, n_days: int) -> list:
    rows = []
    for p0 in range(0, n_days, period_days):
        g = o[(o["day"] >= p0) & (o["day"] < p0 + period_days)]
        if not len(g):
            continue
        s = summarize(g)
        s["day0"] = p0
        rows.append(s)
    return rows

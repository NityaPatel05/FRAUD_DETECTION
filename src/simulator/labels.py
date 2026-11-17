"""
The label process: how ground truth turns into the labels a real fraud team would have.

Gate A  outcome visibility  - a DECLINED row has no outcome (``unobserved``), except a configurable
                              share of declined fraud confirmed through cardholder contact, and rows an
                              analyst reviewed.
Gate B  reporting           - an APPROVED fraud row is reported with a per-scenario probability
                              (card-testing pennies on dormant cards and mule layers are rarely noticed).
Gate C  chargeback          - reported rows become chargebacks after a lognormal delay (median ~38
                              days) that must land inside the label window (120 days); a share is
                              reversed by merchant representment.
Corruption                  - never-reported fraud looks legit at maturity (``unreported``, label 0);
                              friendly fraud: legit purchases disputed as fraud (true=0, reported=1).

``label_mode="oracle"`` short-circuits all gates (every fraud row is confirmed after a short delay)
and reproduces the old simulator behaviour for comparison.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig

DAY = 86_400

STATUSES = ["no_dispute", "unobserved", "unreported", "confirmed", "chargeback", "reported", "reversed",
            "review_confirmed", "review_cleared"]

DISPUTE_REASON = {
    "stolen_card_cnp": "fraud_cnp", "card_testing": "fraud_cnp", "transaction_laundering": "fraud_cnp",
    "merchant_collusion": "fraud_cnp", "account_takeover": "fraud_account_takeover",
    "terminal_cashout": "fraud_counterfeit", "skimming": "fraud_counterfeit",
    "mule_ring": "fraud_unauthorized", "mule_victim": "scam_push_payment",
    "synthetic_identity": "first_party_default", "bust_out": "first_party_default",
}


def _lookup(arr: np.ndarray, table: dict, default) -> np.ndarray:
    u, inv = np.unique(np.asarray(arr).astype(str), return_inverse=True)
    vals = np.array([table.get(x, default) for x in u], dtype=object)
    return vals[inv]


def reason_for(scenario: np.ndarray, role: np.ndarray) -> np.ndarray:
    key = np.where(np.asarray(role).astype(str) == "mule_victim", "mule_victim", scenario)
    return _lookup(key, DISPUTE_REASON, "fraud_unauthorized")


def _lognormal_days(rng, median, sigma, n):
    median = np.broadcast_to(np.asarray(median, float), (n,))
    return rng.lognormal(np.log(median), sigma, n)


def assign(rng: np.random.Generator, cfg: SimConfig, *, t: np.ndarray, is_fraud: np.ndarray, scenario: np.ndarray,
           role: np.ndarray, approved: np.ndarray, declined_by: np.ndarray, decision: np.ndarray,
           real_owner: np.ndarray, channel: np.ndarray, txn_type: np.ndarray, reviewed_fraud: np.ndarray,
           reviewed: np.ndarray) -> dict:
    """Label columns for one epoch of decided rows (all arrays aligned, length n).

    Returns arrays: status, reported (float, nan = no outcome), label_t (seconds, nan = never),
    report_t (when the cardholder/analyst learns of the fraud -> triggers a block; nan = never),
    dispute (reason or None)."""
    n = len(t)
    window = cfg.label_window_days * DAY
    status = np.full(n, "no_dispute", dtype=object)
    reported = np.zeros(n)
    label_t = (t + window).astype(float)
    report_t = np.full(n, np.nan)
    fraud = is_fraud == 1
    refund = txn_type == "refund"
    u = rng.random(n)

    if cfg.label_mode == "oracle":
        d = rng.uniform(0.1, 3, n) * DAY
        cb = np.clip(_lognormal_days(rng, cfg.cb_delay_median_days, cfg.cb_delay_sigma, n), 1, cfg.label_window_days)
        card = approved & (channel != "P2P")
        status[fraud] = np.where(card, "chargeback", "confirmed")[fraud]
        reported[fraud] = 1.0
        label_t = np.where(fraud, t + np.where(card, cb * DAY, d), label_t)
        report_t = np.where(fraud & real_owner, t + d, np.nan)
        return dict(status=status, reported=reported, label_t=label_t, report_t=report_t,
                    cb_t=np.where(fraud & card, label_t, np.nan),
                    dispute=np.where(fraud, reason_for(scenario, role), None).astype(object))

    declined = ~approved
    # ── Gate A: declined rows have no outcome ──
    status[declined] = "unobserved"
    reported[declined] = np.nan
    label_t[declined] = np.nan
    contact = declined & fraud & real_owner & (u < cfg.declined_fraud_confirm_share)
    status[contact] = "confirmed"
    reported[contact] = 1.0
    d_contact = rng.uniform(0.1, 3, n) * DAY
    label_t[contact] = (t + d_contact)[contact]
    report_t[contact] = (t + d_contact)[contact]

    # analyst-reviewed rows are labelled by the analyst (next day or two)
    lat = rng.uniform(0.1, 2, n) * DAY
    rv = reviewed & ~refund
    conf = rv & reviewed_fraud
    clr = rv & ~reviewed_fraud
    status[conf] = "review_confirmed"
    reported[conf] = 1.0
    label_t[conf] = (t + lat)[conf]
    report_t[conf] = np.where(real_owner, t + lat, np.nan)[conf]
    status[clr] = "review_cleared"
    reported[clr] = 0.0
    label_t[clr] = (t + lat)[clr]

    # ── Gate B: approved fraud is reported with a per-scenario probability ──
    has_role = np.asarray(role).astype(str) != "None"
    key = np.where(has_role, role, scenario).astype(object)       # a mule role overrides the scenario
    p_rep = _lookup(key, cfg.report_prob, 0.5).astype(float)
    ok = approved & fraud & ~rv
    rep = ok & (u < p_rep)
    # time until the victim notices & contacts the issuer (drives the card block)
    med_rd = _lookup(scenario, cfg.report_delay_days, 7.0).astype(float)
    rd = np.clip(_lognormal_days(rng, med_rd, 0.9, n), 0.05, 60.0) * DAY
    # ── Gate C: chargeback delay inside the window ──
    med_cb = _lookup(scenario, cfg.cb_median_by_scenario, cfg.cb_delay_median_days).astype(float)
    cb = _lognormal_days(rng, med_cb, cfg.cb_delay_sigma, n) * DAY
    cb = np.maximum(cb, rd + 2 * DAY)
    in_win = cb <= window
    cbk = rep & in_win
    p2p = channel == "P2P"
    cb_t = np.where(cbk, t + cb, np.nan)
    status[cbk] = np.where(p2p, "reported", np.where(refund, "confirmed", "chargeback"))[cbk]
    reported[cbk] = 1.0
    label_t[cbk] = (t + cb)[cbk]
    rev = cbk & ~p2p & ~refund & (rng.random(n) < cfg.representment_rate)
    status[rev] = "reversed"
    reported[rev] = 0.0
    label_t[rev] = (t + cb + rng.uniform(10, 40, n) * DAY)[rev]
    # reported by the victim (so the card is blocked) even if the chargeback arrives too late
    report_t[rep] = (t + rd)[rep]
    status[ok & ~cbk] = "unreported"           # never reported, or chargeback outside the window
    # (reported=0, label_t=maturity stay as initialised: looks legit once the window has passed)

    dispute = np.where(reported == 1, reason_for(scenario, role), None).astype(object)
    dispute[rev] = reason_for(scenario, role)[rev]
    return dict(status=status, reported=reported, label_t=label_t, report_t=report_t, cb_t=cb_t, dispute=dispute)


def friendly_probability(cfg: SimConfig, n_true_disputes: float, n_eligible: float) -> float:
    """P(an eligible legit purchase is disputed) so that friendly / all disputes ~= friendly_share."""
    f = cfg.friendly_share_of_disputes
    if f <= 0 or n_eligible <= 0:
        return 0.0
    prior_n = 20_000.0
    prior_rate = 0.6 * cfg.target_fraud_rate
    ratio = (n_true_disputes + prior_n * prior_rate) / (n_eligible + prior_n)
    return float(min(f / (1 - f) * ratio, 0.05))

"""
Real-world feature set for the SIMULATED payments world (Track C).

The synthetic world has the fields production systems have and IEEE-CIS lacks
(customer / card / merchant / terminal / device / IP ids, geo, AVS/CVC/3DS, auth codes,
real label arrival dates). These are the features the industry research points to —
all CAUSAL (only earlier rows; labels only once they had actually arrived):

  velocity          counts / amounts per card, customer, device, IP, merchant, terminal, BIN
  prior declines    declined attempts per card / device / IP before this one (card testing)
  deviation         amount vs the customer's own past (expanding + decayed EMA)
  novelty           first time this customer uses this device / IP / merchant / IP-country
  sharing (graph)   cards & customers per device / IP, senders per P2P receiver (mule fan-in)
  impossible travel km/h from the customer's previous geo-located transaction
  mismatch flags    card vs IP vs billing vs shipping vs merchant country; AVS/CVC/3DS codes
  label history     frauds reported (label_date <= now) per customer, card, device, IP,
                    merchant, terminal → terminal/merchant risk (Fraud Detection Handbook)
  identity clusters customers linked through shared devices (hub-capped union-find)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.feature_engineering.entity_ops import (
    _Sorted, entity_count, entity_mean_std, first_time_pair, make_codes, prev_row_values,
    smoothed_rate, window_count_sum,
)
from src.feature_engineering.streaming_ops import decayed_profile, identity_clusters, window_distinct

GROUND_TRUTH = ["is_fraud", "is_fraud_true", "reported_label", "label_status", "label_date", "chargeback_date",
                "dispute_reason", "fraud_scenario", "incident_id", "ring_id", "gang_id"]
OUTCOMES = ["auth_response_code", "approved", "decision", "risk_score", "propensity", "declined_by", "liability_shift"]
MATURITY_DAYS = 90                      # = src.labels.policy.DEFAULT_POLICY.maturity_days
BURST_PREFIX = "burst_"                 # card-testing / attack features (kept out of the general feature set)
IDS = ["transaction_id", "customer_id", "card_id", "merchant_id", "terminal_id", "device_id",
       "address_id", "counterparty_customer_id", "ip"]
CATEGORICAL = ["email_domain", "billing_country", "card_country", "card_type", "card_network", "mcc",
               "merchant_country", "channel", "pos_entry_mode", "currency", "ip_country", "ip_type",
               "device_os", "browser", "shipping_country", "avs_result", "cvc_check", "three_ds_result"]
ENTITIES = ["card_id", "customer_id", "device_id", "ip", "merchant_id", "terminal_id", "card_bin"]


def _f32(x):
    return np.asarray(x, dtype=np.float32)


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def label_arrival_counts(codes: np.ndarray, t: np.ndarray, label_t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """# fraud labels of the entity's transactions that had ARRIVED (label_t < t) before row time t.
    Uses the real label_date of each fraud row, so the row's own label (which arrives
    later) can never be counted."""
    s = _Sorted(codes, t)
    fraud = (np.asarray(y) == 1) & (codes >= 0) & ~np.isnan(label_t)
    lc = codes[fraud].astype(np.int64)
    lt = label_t[fraud].astype(np.int64)
    stride = np.int64(1_000_000_000)
    keys = np.sort(lc * stride + lt)
    tx_key = s.c * stride + s.t
    hi = np.searchsorted(keys, tx_key, side="left")               # arrivals strictly before t
    lo = np.searchsorted(keys, s.c * stride, side="left")
    return s.to_orig((hi - lo).astype(np.float64))


def ready_counts(codes: np.ndarray, t: np.ndarray, mask: np.ndarray, ready_t: np.ndarray) -> np.ndarray:
    """# rows of the same entity with ``mask`` whose information is READY (ready_t < t) before row time t."""
    s = _Sorted(codes, t)
    ok = mask & (codes >= 0) & ~np.isnan(ready_t)
    stride = np.int64(1_000_000_000)
    keys = np.sort(codes[ok].astype(np.int64) * stride + ready_t[ok].astype(np.int64))
    tx_key = s.c * stride + s.t
    hi = np.searchsorted(keys, tx_key, side="left")
    lo = np.searchsorted(keys, s.c * stride, side="left")
    return s.to_orig((hi - lo).astype(np.float64))


_ES = np.int64(1 << 28)


def _ev_keys(code: np.ndarray, t: np.ndarray) -> np.ndarray:
    return code.astype(np.int64) * _ES + (t.astype(np.int64) + 10_000_000)


def sec_since_last(ev_code, ev_t, tx_code, tx_t) -> np.ndarray:
    """Seconds from the latest earlier event with the same key (NaN if none / key missing)."""
    keys = np.sort(_ev_keys(ev_code, ev_t))
    q = _ev_keys(tx_code, tx_t)
    pos = np.searchsorted(keys, q, side="left") - 1
    ok = (pos >= 0) & (tx_code >= 0)
    last = keys[np.clip(pos, 0, None)] if len(keys) else q
    ok &= (last // _ES) == tx_code
    return np.where(ok, (q - last).astype(np.float64), np.nan)


def count_events(ev_code, ev_t, tx_code, tx_t, window: int) -> np.ndarray:
    keys = np.sort(_ev_keys(ev_code, ev_t))
    q = _ev_keys(tx_code, tx_t)
    n = np.searchsorted(keys, q, side="left") - np.searchsorted(keys, q - window, side="left")
    return np.where(tx_code >= 0, n.astype(np.float64), np.nan)


def event_features(df: pd.DataFrame, events: pd.DataFrame, t0, t: np.ndarray) -> Dict[str, np.ndarray]:
    """Account-event features (login / credential change / device enrolment ...) strictly BEFORE each txn."""
    ev = events.sort_values("event_time", kind="mergesort")
    et = ((ev["event_time"] - t0).dt.total_seconds()).to_numpy().astype(np.int64)
    cust_all = pd.concat([df["customer_id"], ev["customer_id"]], ignore_index=True).astype("float64")
    cc, _ = pd.factorize(cust_all)
    ccx, cct = cc[len(df):], cc[: len(df)]
    etype = ev["event_type"].to_numpy()
    succ = ev["success"].to_numpy().astype(bool)
    kinds = {"login_ok": (etype == "login") & succ, "login_fail": (etype == "login") & ~succ}
    for k in ["password_reset", "email_change", "phone_change", "device_enrolment", "new_payee"]:
        kinds[k] = etype == k
    out: Dict[str, np.ndarray] = {}
    for name, m in kinds.items():
        out[f"acct_sec_since_{name}"] = sec_since_last(ccx[m], et[m], cct, t)
    cred = kinds["password_reset"] | kinds["email_change"] | kinds["phone_change"]
    bad = kinds["login_fail"]
    out["acct_sec_since_cred_change"] = sec_since_last(ccx[cred], et[cred], cct, t)
    for w, tag in [(86400, "1d"), (7 * 86400, "7d")]:
        out[f"acct_failed_logins_{tag}"] = count_events(ccx[bad], et[bad], cct, t, w)
        out[f"acct_cred_changes_{tag}"] = count_events(ccx[cred], et[cred], cct, t, w)
    ic, _ = pd.factorize(pd.concat([df["ip"], ev["ip"]], ignore_index=True))
    icx, ict = ic[len(df):], ic[: len(df)]
    for w, tag in [(3600, "1h"), (86400, "1d")]:
        out[f"acct_ip_failed_logins_{tag}"] = count_events(icx[bad], et[bad], ict, t, w)
    # was this (customer, device) pair ever seen in the account-event stream before?
    ca = cust_all.to_numpy()
    dv = pd.concat([df["device_id"], ev["device_id"]], ignore_index=True).astype("float64").to_numpy()
    pair = np.where(np.isnan(dv) | np.isnan(ca), -1.0, ca * 1_000_003.0 + dv % 1_000_003)
    pc, _ = pd.factorize(pair)
    pc = np.where(pair < 0, -1, pc)
    pcx, pct = pc[len(df):], pc[: len(df)]
    seen = sec_since_last(pcx, et, pct, t)
    out["acct_sec_since_device_event"] = seen
    out["acct_device_unseen_in_events"] = np.where(pct < 0, np.nan, np.isnan(seen).astype(float))
    return {k: _f32(v) for k, v in out.items()}


def burst_features(df: pd.DataFrame, codes: Dict[str, np.ndarray], t: np.ndarray, amt: np.ndarray,
                   small_usd: float = 5.0) -> Dict[str, np.ndarray]:
    """Attack-burst features (card testing): many tiny auths / many distinct cards / decline spikes on
    one merchant, BIN, terminal, IP or device within minutes. Causal (windows end at the row)."""
    out: Dict[str, np.ndarray] = {}
    P = BURST_PREFIX
    small = (amt <= small_usd).astype(np.float64)
    declined = (df["approved"].to_numpy() == 0).astype(np.float64)
    for e in ["merchant_id", "card_bin", "terminal_id", "ip", "device_id"]:
        c = codes[e]
        for w, tag in [(300, "5m"), (3600, "60m")]:
            n = window_count_sum(c, t, w)["count"]
            rs = window_count_sum(c, t, w, values=small)["sum"]
            rd = window_count_sum(c, t, w, values=declined)["sum"]
            out[f"{P}{e}_n_{tag}"] = _f32(n)
            out[f"{P}{e}_small_n_{tag}"] = _f32(rs)
            out[f"{P}{e}_small_frac_{tag}"] = _f32(rs / np.maximum(n, 1))
            pn = n - 1
            out[f"{P}{e}_decline_rate_{tag}"] = _f32(
                np.where(pn > 0, (rd - np.where(c >= 0, declined, np.nan)) / np.maximum(pn, 1), np.nan))
            if e != "device_id":
                dc = window_distinct(c, t, codes["card_id"], w)
                out[f"{P}{e}_distinct_cards_{tag}"] = _f32(dc)
                if tag == "60m":
                    out[f"{P}{e}_cards_per_txn_60m"] = _f32(dc / np.maximum(n, 1))
    m7 = window_count_sum(codes["merchant_id"], t, 7 * 86400)["count"]
    out[f"{P}merchant_surge_vs_7d"] = _f32(out[f"{P}merchant_id_n_60m"] / (m7 / 168.0 + 1.0))
    out[f"{P}card_distinct_merchants_1h"] = _f32(window_distinct(codes["card_id"], t, codes["merchant_id"], 3600))
    out[f"{P}card_small_n_1h"] = _f32(window_count_sum(codes["card_id"], t, 3600, values=small)["sum"])
    out[f"{P}is_small"] = _f32(small)
    return out


def build_sim_features(df: pd.DataFrame, events: pd.DataFrame | None = None, cfg: Dict | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """df must be the simulator's transactions (time-sorted). Returns (features, feature_cols)."""
    cfg = {"hub_cap": 20, "smoothing": 20.0, "cluster_label_delay_days": 30, "base_rate": 0.01, **(cfg or {})}
    df = df.sort_values(["event_time", "transaction_id"], kind="mergesort").reset_index(drop=True)
    t0 = df["event_time"].min()
    t = ((df["event_time"] - t0).dt.total_seconds()).to_numpy().astype(np.int64)
    lab_t = ((df["label_date"] - t0).dt.total_seconds()).to_numpy(dtype=np.float64)
    # LABELS AS DELIVERED: reported_label (never is_fraud_true); a row's label exists only from its label_date on
    y = df["reported_label"].fillna(0).to_numpy().astype(np.int64)
    observed = df["label_date"].notna().to_numpy()
    amt = df["amount_usd"].to_numpy(dtype=np.float64)
    out: Dict[str, np.ndarray] = {}
    codes = {e: make_codes(df, e) for e in ENTITIES if e in df.columns}

    # row-level
    out["amount_usd"] = _f32(amt); out["log_amount"] = _f32(np.log1p(amt))
    out["local_hour"] = _f32(df["local_hour"]); out["account_age_days"] = _f32(df["account_age_days"])
    out["merchant_age_days"] = _f32(df["merchant_age_days"]); out["is_refund"] = _f32(df["txn_type"] == "refund")
    out["recurring"] = _f32(df["recurring"])
    out["amount_is_round"] = _f32(np.abs(df["amount"] - np.round(df["amount"])) < 1e-9)
    for a, b in [("card_country", "ip_country"), ("billing_country", "ip_country"), ("shipping_country", "billing_country"),
                 ("merchant_country", "card_country"), ("card_country", "billing_country")]:
        both = df[a].notna() & df[b].notna()
        out[f"{a}_ne_{b}"] = _f32(np.where(both, (df[a] != df[b]).astype(float), np.nan))
    for c in CATEGORICAL:
        cc, _ = pd.factorize(df[c].astype("string"), sort=True)
        out[c] = _f32(np.where(cc < 0, np.nan, cc))

    # prior declines (outcomes of EARLIER rows only: window includes current → subtract it)
    declined = (df["approved"].to_numpy() == 0).astype(np.float64)
    for e in ["card_id", "device_id", "ip", "card_bin", "merchant_id"]:
        for w, tag in [(3600, "1h"), (86400, "1d")]:
            r = window_count_sum(codes[e], t, w, values=declined)
            out[f"{e}_prior_declines_{tag}"] = _f32(r["sum"] - np.where(codes[e] >= 0, declined, np.nan))

    # velocity
    for e in ENTITIES:
        for w, tag in [(600, "10m"), (3600, "1h"), (86400, "1d"), (7 * 86400, "7d"), (30 * 86400, "30d")]:
            if (e in ("merchant_id", "terminal_id", "card_bin") and w == 600) or (
                    w == 30 * 86400 and e in ("device_id", "ip", "terminal_id", "card_bin")):
                continue
            r = window_count_sum(codes[e], t, w, values=amt)
            out[f"{e}_n_{tag}"] = _f32(r["count"]); out[f"{e}_amt_{tag}"] = _f32(r["sum"])
        out[f"{e}_n_total"] = _f32(entity_count(codes[e], t, "causal"))

    # deviation from the customer's own history
    cust = codes["customer_id"]
    prior = entity_mean_std(cust, t, amt, "causal", include_current=False)
    out["amt_z_vs_customer"] = _f32(np.clip((amt - prior["mean"]) / np.maximum(np.nan_to_num(prior["std"]), 1.0), -10, 10))
    out["amt_ratio_vs_customer_max"] = _f32(amt / np.maximum(prior["mean"], 0.01))
    for hl, tag in [(86400, "1d"), (30 * 86400, "30d")]:
        p = decayed_profile(cust, t, amt, hl)
        out[f"cust_ema_n_{tag}"] = _f32(p["count"])
        out[f"cust_ema_z_{tag}"] = _f32(np.clip((amt - p["mean"]) / np.maximum(np.nan_to_num(p["std"]), 1.0), -10, 10))

    # novelty for the customer / card
    for ent, val in [("customer_id", "device_id"), ("customer_id", "ip"), ("customer_id", "merchant_id"),
                     ("customer_id", "ip_country"), ("card_id", "merchant_id"), ("card_id", "device_id"), ("customer_id", "mcc"),
                     ("customer_id", "merchant_country")]:
        out[f"{ent}_new_{val}"] = _f32(first_time_pair(codes.get(ent, make_codes(df, ent)), t,
                                                         codes.get(val, make_codes(df, val))))

    # sharing / graph
    for ent, nb, w, tag in [("device_id", "card_id", 86400, "1d"), ("device_id", "card_id", 7 * 86400, "7d"),
                            ("device_id", "customer_id", 7 * 86400, "7d"), ("ip", "card_id", 86400, "1d"),
                            ("ip", "customer_id", 86400, "1d"), ("customer_id", "device_id", 30 * 86400, "30d"),
                            ("card_bin", "card_id", 3600, "1h"), ("merchant_id", "card_id", 3600, "1h")]:
        out[f"{ent}_distinct_{nb}_{tag}"] = _f32(window_distinct(codes[ent], t, codes[nb], w))
    recv = make_codes(df, "counterparty_customer_id")
    out["receiver_distinct_senders_7d"] = _f32(window_distinct(recv, t, cust, 7 * 86400))
    out["receiver_n_inbound_1d"] = _f32(window_count_sum(recv, t, 86400)["count"])
    out["sender_account_age_lt_30d"] = _f32(df["account_age_days"] < 30)

    # merchant health: volume ramp, refund ratio, decline rate (collusion / laundering / bust-out merchants)
    mc = codes["merchant_id"]
    refund = (df["txn_type"] == "refund").to_numpy().astype(np.float64)
    for w, tag in [(86400, "1d"), (7 * 86400, "7d"), (30 * 86400, "30d")]:
        out[f"merchant_refund_n_{tag}"] = _f32(window_count_sum(mc, t, w, values=refund)["sum"])
        out[f"merchant_refund_ratio_{tag}"] = _f32(out[f"merchant_refund_n_{tag}"] / np.maximum(out[f"merchant_id_n_{tag}"], 1))
    out["merchant_ramp_1d_vs_30d"] = _f32(out["merchant_id_n_1d"] / (out["merchant_id_n_30d"] / 30.0 + 1.0))
    out["merchant_ramp_7d_vs_30d"] = _f32(out["merchant_id_n_7d"] / 7.0 / (out["merchant_id_n_30d"] / 30.0 + 0.1))
    out["merchant_distinct_cards_30d"] = _f32(window_distinct(mc, t, codes["card_id"], 30 * 86400))
    out["merchant_decline_rate_1d"] = _f32(out["merchant_id_prior_declines_1d"] / np.maximum(out["merchant_id_n_1d"] - 1, 1))

    # impossible travel (customer's previous geolocated row)
    lat, lon = df["ip_lat"].to_numpy(dtype=np.float64), df["ip_lon"].to_numpy(dtype=np.float64)
    geo_cust = np.where(np.isnan(lat), -1, cust)                     # only rows with a location
    plat, plon = prev_row_values(geo_cust, t, lat), prev_row_values(geo_cust, t, lon)
    pt = prev_row_values(geo_cust, t, t.astype(np.float64))
    km = _haversine_km(plat, plon, lat, lon)
    hours = np.maximum((t - pt) / 3600.0, 1 / 60)
    out["km_from_prev_geo"] = _f32(km); out["kmh_from_prev_geo"] = _f32(km / hours)
    out["impossible_travel"] = _f32(np.where(np.isnan(km), np.nan, (km / hours > 900).astype(float)))

    # label history with REAL label arrival times (terminal / merchant risk etc.)
    # fraud = REPORTED fraud whose label_date < now; rate denominator = prior rows that are MATURE (t+90d < now),
    # numerator = reported fraud among those (policy.py semantics: same population, no early-report inflation)
    mat = float(MATURITY_DAYS * 86400)
    is_pos = (y == 1) & ~np.isnan(lab_t)
    for e in ["customer_id", "card_id", "device_id", "ip", "merchant_id", "terminal_id", "card_bin"]:
        nf = ready_counts(codes[e], t, is_pos, lab_t)
        nmat = ready_counts(codes[e], t, observed, t + mat)
        fmat = ready_counts(codes[e], t, is_pos, np.maximum(lab_t, t + mat))
        out[f"{e}_fraud_reported"] = _f32(nf)
        out[f"{e}_n_mature"] = _f32(nmat)
        out[f"{e}_fraud_rate"] = _f32(smoothed_rate(fmat, nmat, cfg["base_rate"], cfg["smoothing"]))
        if e == "merchant_id":                      # chargeback-rate trajectory: recent reports vs mature base
            out["merchant_fraud_recent"] = _f32(np.clip(nf - fmat, 0, None))

    # identity clusters through shared devices (hub-capped, labels after a delay)
    dev = codes["device_id"]
    newp = first_time_pair(dev, t, cust)
    newp = np.where(np.isnan(newp), np.where(dev >= 0, 1.0, 0.0), newp)
    d_lab = lab_t - t                                              # only labels that ARRIVE within the delay count
    y_cl = ((y == 1) & (d_lab <= cfg["cluster_label_delay_days"] * 86400)).astype(float)
    cl = identity_clusters(t, cust, dev, newp, y_cl, cfg["cluster_label_delay_days"] * 86400, cfg["hub_cap"])
    out["cluster_n_customers"] = _f32(cl["size"]); out["cluster_fraud"] = _f32(cl["fraud"])
    out["cluster_merges"] = _f32(cl["merge"])

    if events is not None and len(events):
        out.update(event_features(df, events, t0, t))
    out.update(burst_features(df, codes, t, amt))
    feats = pd.DataFrame(out)
    keep = df[["transaction_id", "event_time", "customer_id", "card_id", "merchant_id", "amount_usd", "channel",
               "account_age_days", "decision", "approved", "label_status", "label_date", "counterparty_customer_id"]
              + [c for c in ["is_fraud_true", "reported_label", "fraud_scenario", "ring_id"] if c in df.columns]
              ].reset_index(drop=True).rename(columns={"channel": "m_channel", "amount_usd": "m_amount_usd",
                                                       "account_age_days": "m_account_age_days"})
    logger.info("Simulated-world features: {} columns for {} rows", feats.shape[1], len(feats))
    cols = list(feats.columns)
    return pd.concat([keep, feats], axis=1), [c for c in cols if not c.startswith(BURST_PREFIX)]

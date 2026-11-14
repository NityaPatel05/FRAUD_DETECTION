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

GROUND_TRUTH = ["is_fraud", "label_date", "chargeback_date", "fraud_scenario", "incident_id", "ring_id"]
OUTCOMES = ["auth_response_code", "approved"]
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


def build_sim_features(df: pd.DataFrame, cfg: Dict | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """df must be the simulator's transactions (time-sorted). Returns (features, feature_cols)."""
    cfg = {"hub_cap": 20, "smoothing": 20.0, "cluster_label_delay_days": 30, **(cfg or {})}
    df = df.sort_values(["event_time", "transaction_id"], kind="mergesort").reset_index(drop=True)
    t0 = df["event_time"].min()
    t = ((df["event_time"] - t0).dt.total_seconds()).to_numpy().astype(np.int64)
    lab_t = ((df["label_date"] - t0).dt.total_seconds()).to_numpy(dtype=np.float64)
    y = df["is_fraud"].to_numpy()
    amt = df["amount_usd"].to_numpy(dtype=np.float64)
    out: Dict[str, np.ndarray] = {}
    codes = {e: make_codes(df, e) for e in ENTITIES if e in df.columns}

    # row-level
    out["amount_usd"] = _f32(amt); out["log_amount"] = _f32(np.log1p(amt))
    out["local_hour"] = _f32(df["local_hour"]); out["account_age_days"] = _f32(df["account_age_days"])
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
        for w, tag in [(600, "10m"), (3600, "1h"), (86400, "1d"), (7 * 86400, "7d")]:
            if e in ("merchant_id", "terminal_id", "card_bin") and w == 600:
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
                     ("customer_id", "ip_country"), ("card_id", "merchant_id"), ("card_id", "device_id")]:
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
    for e in ["customer_id", "card_id", "device_id", "ip", "merchant_id", "terminal_id", "card_bin"]:
        nf = label_arrival_counts(codes[e], t, lab_t, y)
        n_prior = out[f"{e}_n_total"] - 1
        out[f"{e}_fraud_reported"] = _f32(nf)
        out[f"{e}_fraud_rate"] = _f32(smoothed_rate(nf, n_prior, 0.035, cfg["smoothing"]))

    # identity clusters through shared devices (hub-capped, labels after a delay)
    dev = codes["device_id"]
    newp = first_time_pair(dev, t, cust)
    newp = np.where(np.isnan(newp), np.where(dev >= 0, 1.0, 0.0), newp)
    cl = identity_clusters(t, cust, dev, newp, y.astype(float), cfg["cluster_label_delay_days"] * 86400, cfg["hub_cap"])
    out["cluster_n_customers"] = _f32(cl["size"]); out["cluster_fraud"] = _f32(cl["fraud"])
    out["cluster_merges"] = _f32(cl["merge"])

    feats = pd.DataFrame(out)
    keep = df[["transaction_id", "event_time"] + GROUND_TRUTH].reset_index(drop=True)
    logger.info("Simulated-world features: {} columns for {} rows", feats.shape[1], len(feats))
    return pd.concat([keep, feats], axis=1), list(feats.columns)

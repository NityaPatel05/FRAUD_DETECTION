"""
Checks for a simulated dataset:

1. fidelity   — marginals vs IEEE-CIS aggregates (fraud rate, amount quantiles,
                share of >2-decimal amounts, hourly curve). Only aggregates are read.
2. leakage    — single-column AUC on a time split (train period -> val+test). No
                single non-ground-truth column should separate fraud on its own.
3. signals    — each scenario must be detectable by the causal signal it was
                designed around (reusing src/feature_engineering primitives), while
                friendly fraud stays ~0.5 (it is label noise by design).
4. structure  — novel scenario only in the test period, labels arrive after the event,
                ground-truth ids only on fraud rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.feature_engineering.entity_ops import delayed_label_stats, prev_row_values, window_count_sum
from src.feature_engineering.streaming_ops import decayed_profile, window_distinct

from . import geo
from .world import GROUND_TRUTH_COLS, SimResult

LEAK_FLAG = 0.85


def _auc(y, x) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, x))


def _decimals_gt2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.abs(x * 100 - np.round(x * 100)) > 1e-6


def amount_profile(amount: np.ndarray, y: np.ndarray) -> dict:
    out = {"fraud_rate": float(np.mean(y))}
    for lab, m in (("legit", y == 0), ("fraud", y == 1)):
        a = amount[m]
        out[lab] = {f"q{int(q * 100)}": float(np.quantile(a, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}
        out[lab]["mean"] = float(a.mean())
        out[lab]["share_gt2_decimals"] = float(_decimals_gt2(a).mean())
    return out


def ieee_reference(train_transaction_csv: str | Path) -> dict:
    """Aggregate statistics only (no rows are kept)."""
    df = pd.read_csv(train_transaction_csv, usecols=["TransactionDT", "TransactionAmt", "isFraud"])
    ref = amount_profile(df["TransactionAmt"].to_numpy(), df["isFraud"].to_numpy())
    hour = (df["TransactionDT"].to_numpy() // 3600) % 24
    ref["hourly_share"] = (np.bincount(hour, minlength=24) / len(hour)).round(4).tolist()
    ref["source"] = str(train_transaction_csv)
    return ref


def leakage_scan(tx: pd.DataFrame, val_start: pd.Timestamp) -> pd.DataFrame:
    y = tx["is_fraud"].to_numpy()
    train = (tx["event_time"] < val_start).to_numpy()
    ev = ~train
    prior = y[train].mean()
    rows = []
    for c in tx.columns:
        if c in GROUND_TRUTH_COLS or c in ("transaction_id", "event_time"):
            continue
        s = tx[c]
        id_like = c.endswith("_id") or c in ("ip", "card_bin")
        if pd.api.types.is_numeric_dtype(s) and not id_like and s.dtype != object:
            x = s.astype(float).to_numpy()
            x = np.where(np.isnan(x), -1e12, x)
            auc = _auc(y[ev], x[ev])
            kind = "numeric"
        else:
            key = s.astype(str).to_numpy()
            g = pd.DataFrame({"k": key[train], "y": y[train]}).groupby("k")["y"].agg(["sum", "count"])
            rate = (g["sum"] + 20 * prior) / (g["count"] + 20)
            x = pd.Series(key[ev]).map(rate).fillna(prior).to_numpy()
            auc = _auc(y[ev], x)
            kind = "categorical (train-period target rate)"
        auc = max(auc, 1 - auc) if not np.isnan(auc) else auc
        rows.append({"column": c, "kind": kind, "auc": round(auc, 4), "flag": bool(auc >= LEAK_FLAG)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def _codes(s: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(s, use_na_sentinel=True)
    return codes.astype(np.int64)


def _new_value_24h(entity: np.ndarray, value: np.ndarray, t: np.ndarray) -> np.ndarray:
    """1 if the entity first used this value < 24h ago AND had history before that.
    Causal: both minima are over rows at or before the current one."""
    df = pd.DataFrame({"e": entity, "v": value, "t": t})
    first_pair = df.groupby(["e", "v"])["t"].transform("min").to_numpy()
    first_ent = df.groupby("e")["t"].transform("min").to_numpy()
    out = ((t - first_pair) < 86_400) & (first_pair > first_ent)
    return np.where((entity < 0) | (value < 0), 0.0, out.astype(float))


def scenario_signals(res: SimResult) -> pd.DataFrame:
    tx = res.transactions
    start = pd.Timestamp(res.metadata["start"])
    t = ((tx["event_time"] - start).dt.total_seconds()).to_numpy().astype(np.int64)
    y = tx["is_fraud"].to_numpy()
    cust = _codes(tx["customer_id"])
    merch = _codes(tx["merchant_id"])
    term = _codes(tx["terminal_id"])
    dev = _codes(tx["device_id"])
    card = _codes(tx["card_id"])
    channel = tx["channel"].to_numpy()

    sig = {}
    c1h = window_count_sum(merch, t, 3600)["count"]
    c7d = window_count_sum(merch, t, 7 * 86_400)["count"]
    sig["merchant_burst_1h"] = c1h / (c7d / 168 + 1)
    sig["new_device_for_customer"] = _new_value_24h(cust, dev, t)
    sig["new_email_for_customer"] = _new_value_24h(cust, _codes(tx["email_domain"]), t)
    sig["ip_vs_card_country_mismatch"] = (tx["ip_country"].notna() & (tx["ip_country"] != tx["card_country"])).to_numpy().astype(float)
    d = delayed_label_stats(term, t, y.astype(float), 7 * 86_400)
    sig["terminal_known_fraud_rate"] = (d["fraud"] + 0.035 * 20) / (d["n"] + 20)

    # card-present travel speed between consecutive card-present uses of the card
    cities = geo.cities_df(geo.countries_df()).set_index("city")
    m_city = res.entities["merchants"].set_index("merchant_id")["city"]
    city = tx["merchant_id"].map(m_city)
    lat = city.map(cities["lat"]).to_numpy(dtype=float)
    lon = city.map(cities["lon"]).to_numpy(dtype=float)
    cp = channel == "CP"
    speed = np.zeros(len(tx))
    idx = np.nonzero(cp)[0]
    plat = prev_row_values(card[idx], t[idx], lat[idx])
    plon = prev_row_values(card[idx], t[idx], lon[idx])
    pt = prev_row_values(card[idx], t[idx], t[idx].astype(float))
    km = geo.haversine_km(plat, plon, lat[idx], lon[idx])
    hours = np.maximum((t[idx] - pt) / 3600, 0.25)
    speed[idx] = np.nan_to_num(km / hours, nan=0.0)
    sig["card_present_travel_kmh"] = speed

    sig["device_distinct_customers_30d"] = np.nan_to_num(window_distinct(dev, t, cust, 30 * 86_400), nan=0.0)
    rcv = _codes(tx["counterparty_customer_id"])
    sig["receiver_distinct_senders_7d"] = np.nan_to_num(window_distinct(rcv, t, cust, 7 * 86_400), nan=0.0)
    la = np.log(tx["amount_usd"].to_numpy())
    prof = decayed_profile(cust, t, la, 30 * 86_400)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (la - prof["mean"]) / np.maximum(prof["std"], 0.25)
    sig["amount_z_vs_own_history"] = np.nan_to_num(z, nan=0.0)

    plan = {
        "card_testing": ["merchant_burst_1h"],
        "stolen_card_cnp": ["new_device_for_customer", "ip_vs_card_country_mismatch"],
        "account_takeover": ["new_device_for_customer", "new_email_for_customer"],
        "terminal_cashout": ["terminal_known_fraud_rate"],
        "skimming": ["card_present_travel_kmh"],
        "mule_ring": ["device_distinct_customers_30d", "receiver_distinct_senders_7d"],
        "bust_out": ["amount_z_vs_own_history"],
        "friendly_fraud": ["amount_z_vs_own_history", "new_device_for_customer"],
    }
    scen = tx["fraud_scenario"].to_numpy()
    rows = []
    for s, names in plan.items():
        m = scen == s
        if not m.any():
            continue
        chans = np.unique(channel[m])
        base = (scen == "legit") & np.isin(channel, chans)
        sel = m | base
        for name in names:
            rows.append({"scenario": s, "signal": name, "channels": "/".join(chans),
                         "n_fraud": int(m.sum()), "auc_vs_legit": round(_auc(m[sel].astype(int), sig[name][sel]), 4)})
    return pd.DataFrame(rows)


def structure_checks(res: SimResult) -> dict:
    tx, meta = res.transactions, res.metadata
    test_start = pd.Timestamp(meta["splits"]["test"][0])
    fr = tx["is_fraud"] == 1
    novel = tx["fraud_scenario"].isin(meta["novel_scenarios"])
    return {
        "novel_only_in_test": bool((tx.loc[novel, "event_time"] >= test_start).all()),
        "labels_after_event": bool((tx.loc[fr, "label_date"] >= tx.loc[fr, "event_time"]).all()),
        "legit_rows_have_no_label_date": bool(tx.loc[~fr, "label_date"].isna().all()),
        "incident_ids_only_on_fraud": bool(tx.loc[~fr, "incident_id"].isna().all()),
        "ring_ids_only_on_fraud": bool(tx.loc[~fr, "ring_id"].isna().all()),
        "chargebacks_only_on_approved": bool((tx.loc[tx["chargeback_date"].notna(), "approved"] == 1).all()),
        "event_time_sorted": bool(tx["event_time"].is_monotonic_increasing),
    }


def summary(res: SimResult) -> dict:
    tx, meta = res.transactions, res.metadata
    sp = meta["splits"]
    out = {"n_rows": int(len(tx)), "fraud_rate": float(tx["is_fraud"].mean())}
    for name, (a, b) in sp.items():
        m = (tx["event_time"] >= pd.Timestamp(a)) & (tx["event_time"] < pd.Timestamp(b))
        out[f"{name}_rows"] = int(m.sum())
        out[f"{name}_fraud_rate"] = float(tx.loc[m, "is_fraud"].mean())
    out["channel_share"] = tx["channel"].value_counts(normalize=True).round(4).to_dict()
    out["device_coverage"] = float(tx["device_id"].notna().mean())
    out["decline_rate_legit"] = float(1 - tx.loc[tx["is_fraud"] == 0, "approved"].mean())
    out["decline_rate_fraud"] = float(1 - tx.loc[tx["is_fraud"] == 1, "approved"].mean())
    g = tx.groupby("fraud_scenario").agg(rows=("is_fraud", "size"), approved=("approved", "mean"),
                                         incidents=("incident_id", "nunique"),
                                         median_usd=("amount_usd", "median"))
    out["scenarios"] = g.round(3).reset_index().to_dict("records")
    out["amounts"] = amount_profile(tx["amount_usd"].to_numpy(), tx["is_fraud"].to_numpy())
    hour = tx["local_hour"].to_numpy().astype(int)
    out["hourly_share_local"] = (np.bincount(hour, minlength=24) / len(hour)).round(4).tolist()
    return out


def validate(res: SimResult, ieee_ref: dict | None = None) -> dict:
    val_start = pd.Timestamp(res.metadata["splits"]["val"][0])
    return {
        "summary": summary(res),
        "structure": structure_checks(res),
        "leakage": leakage_scan(res.transactions, val_start).to_dict("records"),
        "signals": scenario_signals(res).to_dict("records"),
        "ieee_reference": ieee_ref,
    }

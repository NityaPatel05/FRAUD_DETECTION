"""
Checks for a simulated dataset:

1. fidelity     - marginals vs IEEE-CIS aggregates (fraud rate, amount quantiles, share of >2-decimal
                  amounts, hourly curve). Only aggregates are read.
2. leakage      - single-column AUC on a time split (train period -> val+test); no single non-ground-
                  truth column should separate fraud on its own.
3. purity       - NO category value with >= 50 rows may have a smoothed fraud rate > 0.9: the AUC scan
                  cannot see small pure categories (a rare email domain, a null customer id, a decline code).
4. row ceiling  - a gradient-boosted model on CURRENT-ROW fields only: per-scenario AUC should stay
                  below ~0.9 (card testing is allowed to be easy).
5. signals      - each scenario is detectable by the causal signal it was designed around.
6. labels       - label gates behave (declined => unobserved, label_date >= event, delays inside the
                  window, friendly fraud true=0 / reported=1, share of disputes ...).
7. calibration  - target statistic vs real-world source vs simulated value, with tolerance and pass/fail.
8. structure    - novel scenario only in the test period, ids only on fraud rows, reissue linkage ...
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.feature_engineering.entity_ops import delayed_label_stats, prev_row_values, window_count_sum
from src.feature_engineering.streaming_ops import decayed_profile, window_distinct

from . import geo
from .assemble import DECISION_COLUMNS, LABEL_COLUMNS, TRUTH_COLUMNS
from .world import SimResult

LEAK_FLAG = 0.85
LEAK_FLAG_ID = 0.95      # ids: gang device/IP reuse is a realistic reputation signal, only a near-perfect blacklist is a leak
PURITY_MIN_ROWS = 50
PURITY_MAX_RATE = 0.9
CEILING_MAX_AUC = 0.95   # per scenario (n >= CEILING_MIN_N); the design target for ATO / stolen card is ~0.93
CEILING_MIN_N = 25
POLICY_COLS = {"decision", "risk_score", "propensity", "liability_shift"}
SKIP_COLS = set(TRUTH_COLUMNS) | set(LABEL_COLUMNS) | POLICY_COLS | {"transaction_id", "event_time",
                                                                        "original_transaction_id"}


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


# ───────────────────────── leakage & purity ─────────────────────────
def leakage_scan(tx: pd.DataFrame, val_start: pd.Timestamp) -> pd.DataFrame:
    y = tx["is_fraud_true"].to_numpy()
    train = (tx["event_time"] < val_start).to_numpy()
    ev = ~train
    prior = y[train].mean()
    rows = []
    for c in tx.columns:
        if c in SKIP_COLS:
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
            x = pd.Series(key[ev]).map(rate)
            if id_like:
                # identifiers: an id never seen in the train period carries no target-rate information, and
                # "unseen" is itself just novelty (a legitimate behavioural signal, covered by the row-only
                # ceiling), so score only the eval rows whose id was seen in training.
                seen = x.notna().to_numpy()
                if seen.sum() == 0 or y[ev][seen].sum() < 5 or y[ev][seen].sum() == seen.sum():
                    auc = 0.5
                else:
                    auc = _auc(y[ev][seen], x.to_numpy()[seen])
                kind = "id (seen-in-train rows, train-period target rate)"
            else:
                auc = _auc(y[ev], x.fillna(prior).to_numpy())
                kind = "categorical (train-period target rate)"
        auc = max(auc, 1 - auc) if not np.isnan(auc) else auc
        rows.append({"column": c, "kind": kind, "auc": round(auc, 4),
                     "flag": bool(auc >= (LEAK_FLAG_ID if id_like else LEAK_FLAG))})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def purity_check(tx: pd.DataFrame, min_rows: int = PURITY_MIN_ROWS, max_rate: float = PURITY_MAX_RATE) -> pd.DataFrame:
    """Every (column, value) with >= ``min_rows`` rows and smoothed fraud rate > ``max_rate``.
    Empty result = pass. Null is a value too (``<null>``). Covers every non-id, non-label column,
    numeric ones through their distinct values when low-cardinality."""
    y = tx["is_fraud_true"].to_numpy()
    prior = float(y.mean())
    out = []
    for c in tx.columns:
        if c in SKIP_COLS:
            continue
        s = tx[c]
        if c.endswith("_id") or c in ("ip", "card_bin", "amount", "amount_usd", "ip_lat", "ip_lon"):
            continue
        if pd.api.types.is_float_dtype(s.dtype) and s.nunique() > 100:
            continue
        key = s.astype(object).where(s.notna(), "<null>").astype(str)
        g = pd.DataFrame({"v": key.to_numpy(), "y": y}).groupby("v")["y"].agg(["sum", "count"])
        g = g[g["count"] >= min_rows]
        rate = (g["sum"] + 20 * prior) / (g["count"] + 20)
        bad = rate[rate > max_rate]
        for v, r in bad.items():
            out.append({"column": c, "value": v, "rows": int(g.loc[v, "count"]), "fraud_rate_smoothed": round(float(r), 4)})
    return pd.DataFrame(out, columns=["column", "value", "rows", "fraud_rate_smoothed"])


ROW_CAT = ["email_domain", "billing_country", "card_country", "card_type", "card_network", "mcc", "merchant_country",
           "channel", "pos_entry_mode", "txn_type", "currency", "ip_country", "ip_type", "device_os", "browser",
           "shipping_country", "avs_result", "cvc_check", "three_ds_result"]
ROW_NUM = ["local_hour", "account_age_days", "merchant_age_days", "amount_usd", "recurring"]


def _row_features(tx: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X = pd.DataFrame(index=tx.index)
    for c in ROW_CAT:
        X[c] = pd.factorize(tx[c].astype(object).where(tx[c].notna(), "<null>"))[0]
    for c in ROW_NUM:
        X[c] = tx[c].astype(float)
    X["ip_card_mismatch"] = (tx["ip_country"].notna() & (tx["ip_country"] != tx["card_country"])).astype(int)
    X["ship_card_mismatch"] = (tx["shipping_country"].notna() & (tx["shipping_country"] != tx["card_country"])).astype(int)
    X["has_device"] = tx["device_id"].notna().astype(int)
    X["has_customer"] = tx["customer_id"].notna().astype(int)
    a = tx["amount_usd"].to_numpy()
    X["round_amount"] = ((np.abs(a - np.round(a)) < 1e-6) | (np.abs(a - np.round(a) + 0.01) < 1e-6)).astype(int)
    return X, ROW_CAT


def row_only_ceiling(tx: pd.DataFrame, val_start: pd.Timestamp, max_legit: int = 300_000, seed: int = 0) -> dict:
    """HistGBDT on current-row fields only (no history, no outcomes): trained before ``val_start`` on
    ``is_fraud_true``, scored per scenario on the later rows against legit rows."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, cat = _row_features(tx)
    y = tx["is_fraud_true"].to_numpy()
    train = (tx["event_time"] < val_start).to_numpy()
    rng = np.random.default_rng(seed)
    tr = np.nonzero(train)[0]
    legit_tr = tr[y[tr] == 0]
    if len(legit_tr) > max_legit:
        legit_tr = rng.choice(legit_tr, max_legit, replace=False)
    sel = np.concatenate([legit_tr, tr[y[tr] == 1]])
    cat_mask = [c in cat for c in X.columns]
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20,
                                         categorical_features=cat_mask, random_state=seed)
    clf.fit(X.iloc[sel], y[sel])
    ev = ~train
    p = clf.predict_proba(X.loc[ev])[:, 1]
    scen = tx.loc[ev, "fraud_scenario"].to_numpy()
    yt = y[ev]
    neg = scen == "legit"
    out = {"overall_auc": _auc(yt, p), "overall_auc_pr": float(average_precision_score(yt, p)) if yt.max() else float("nan"),
           "per_scenario": {}}
    for s in sorted(set(scen) - {"legit", "synthetic_identity_build"}):
        m = scen == s
        if s == "friendly_fraud" or m.sum() < 12:
            continue
        sub = m | neg
        out["per_scenario"][s] = {"n": int(m.sum()), "auc": round(_auc(m[sub].astype(int), p[sub]), 4)}
    out["pass"] = all(v["auc"] < CEILING_MAX_AUC for k, v in out["per_scenario"].items()
                      if k not in ("card_testing", "skimming") and v["n"] >= CEILING_MIN_N)
    return out


# ───────────────────────── scenario signals ─────────────────────────
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
    y = tx["is_fraud_true"].to_numpy()
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
    sig["terminal_known_fraud_rate"] = (d["fraud"] + 0.01 * 20) / (d["n"] + 20)
    sig["young_merchant"] = -np.nan_to_num(tx["merchant_age_days"].astype(float).to_numpy(), nan=1e4)

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
        "synthetic_identity": ["amount_z_vs_own_history"],
        "merchant_collusion": ["young_merchant", "merchant_burst_1h"],
        "bust_out": ["amount_z_vs_own_history"],
        "friendly_fraud": ["amount_z_vs_own_history", "new_device_for_customer"],
    }
    scen = tx["fraud_scenario"].to_numpy()
    rows = []
    for s, names in plan.items():
        m = scen == s
        if m.sum() < 5:
            continue
        chans = np.unique(channel[m])
        base = (scen == "legit") & np.isin(channel, chans)
        sel = m | base
        for name in names:
            rows.append({"scenario": s, "signal": name, "channels": "/".join(chans),
                         "n_fraud": int(m.sum()), "auc_vs_legit": round(_auc(m[sel].astype(int), sig[name][sel]), 4)})
    return pd.DataFrame(rows)


# ───────────────────────── labels, structure, calibration ─────────────────────────
def label_checks(res: SimResult) -> dict:
    tx, cfg = res.transactions, res.metadata["config"]
    win = cfg["label_window_days"]
    d = (tx["label_date"] - tx["event_time"]).dt.total_seconds() / 86_400
    has = tx["label_date"].notna()
    st = tx["label_status"]
    fr = tx["is_fraud_true"] == 1
    declined = tx["approved"] == 0
    fraud_declined = declined & fr
    cb = tx["label_status"].isin(["chargeback"])
    disputes = tx["reported_label"].eq(1) & st.isin(["chargeback", "reported", "reversed"])
    friendly = disputes & ~fr
    out = {
        "unobserved_has_no_label": bool(tx.loc[st == "unobserved", "reported_label"].isna().all()
                                        and tx.loc[st == "unobserved", "label_date"].isna().all()),
        "label_date_not_before_event": bool((d[has] >= 0).all()),
        "declined_legit_all_unobserved_or_reviewed": bool(
            tx.loc[declined & ~fr, "label_status"].isin(["unobserved", "review_confirmed", "review_cleared"]).all()),
        "share_declined_fraud_with_outcome": float((fraud_declined & (st != "unobserved")).sum() / max(fraud_declined.sum(), 1)),
        "chargeback_delay_median_days": float(d[cb].median()) if cb.any() else float("nan"),
        "chargeback_delay_max_days": float(d[cb].max()) if cb.any() else float("nan"),
        "chargebacks_inside_window": bool((d[cb] <= win + 1e-6).all()) if cb.any() else True,
        "friendly_rows_true0_reported1": bool(((tx["fraud_scenario"] == "friendly_fraud") <= (~fr & tx["reported_label"].eq(1))).all()),
        "friendly_share_of_disputes": float(friendly.sum() / max(disputes.sum(), 1)),
        "share_fraud_eventually_reported_1": float((fr & tx["reported_label"].eq(1)).sum() / max(fr.sum(), 1)),
        "share_fraud_unreported": float((fr & (st == "unreported")).sum() / max(fr.sum(), 1)),
        "status_counts": st.value_counts().to_dict(),
    }
    return out


def structure_checks(res: SimResult) -> dict:
    tx, meta = res.transactions, res.metadata
    test_start = pd.Timestamp(meta["splits"]["test"][0])
    fr = tx["is_fraud_true"] == 1
    novel = tx["fraud_scenario"].isin(meta["novel_scenarios"])
    cards = res.entities["cards"]
    reissued = cards[cards["prev_card_id"].notna()]
    prev_ok = bool(reissued["prev_card_id"].isin(cards["card_id"]).all())
    blocked_prev = cards.set_index("card_id")["blocked_date"].reindex(reissued["prev_card_id"]).notna().all()
    return {
        "novel_only_in_test": bool((tx.loc[novel, "event_time"] >= test_start).all()),
        "is_fraud_equals_is_fraud_true": bool((tx["is_fraud"] == tx["is_fraud_true"]).all()),
        "incident_ids_only_on_fraud_or_build": bool(tx.loc[~fr & (tx["fraud_scenario"] == "legit"), "incident_id"].isna().all()),
        "ring_ids_only_on_fraud": bool(tx.loc[~fr, "ring_id"].isna().all()),
        "chargebacks_only_on_approved": bool((tx.loc[tx["chargeback_date"].notna(), "approved"] == 1).all()),
        "event_time_sorted": bool(tx["event_time"].is_monotonic_increasing),
        "transaction_ids_unique": bool(tx["transaction_id"].is_unique),
        "reissue_links_valid": prev_ok and bool(blocked_prev),
        "declined_rows_have_no_chargeback": bool(tx.loc[tx["approved"] == 0, "chargeback_date"].isna().all()),
    }


def _fraud_value_rates(tx: pd.DataFrame) -> dict:
    card = tx[(tx["channel"] != "P2P") & (tx["txn_type"] == "purchase") & (tx["approved"] == 1)]
    f = card["is_fraud_true"] == 1
    v_f = card.loc[f, "amount_usd"].sum()
    return {"fraud_value_rate": float(v_f / max(card["amount_usd"].sum(), 1.0)),
            "cnp_share_of_fraud_value": float(card.loc[f & (card["channel"] == "CNP"), "amount_usd"].sum() / max(v_f, 1.0))}


def summary(res: SimResult) -> dict:
    tx, meta = res.transactions, res.metadata
    sp = meta["splits"]
    out = {"n_rows": int(len(tx)), "fraud_rate": float(tx["is_fraud_true"].mean())}
    for name, (a, b) in sp.items():
        m = (tx["event_time"] >= pd.Timestamp(a)) & (tx["event_time"] < pd.Timestamp(b))
        out[f"{name}_rows"] = int(m.sum())
        out[f"{name}_fraud_rate"] = float(tx.loc[m, "is_fraud_true"].mean())
    out.update(_fraud_value_rates(tx))
    out["channel_share"] = tx["channel"].value_counts(normalize=True).round(4).to_dict()
    out["device_coverage"] = float(tx["device_id"].notna().mean())
    out["decline_rate_legit"] = float(1 - tx.loc[tx["is_fraud_true"] == 0, "approved"].mean())
    out["decline_rate_fraud"] = float(1 - tx.loc[tx["is_fraud_true"] == 1, "approved"].mean())
    out["decision_share"] = tx["decision"].value_counts(normalize=True).round(4).to_dict()
    g = tx.groupby("fraud_scenario").agg(rows=("is_fraud_true", "size"), fraud=("is_fraud_true", "sum"),
                                         approved=("approved", "mean"), incidents=("incident_id", "nunique"),
                                         median_usd=("amount_usd", "median"))
    out["scenarios"] = g.round(3).reset_index().to_dict("records")
    out["amounts"] = amount_profile(tx["amount_usd"].to_numpy(), tx["is_fraud_true"].to_numpy())
    hour = tx["local_hour"].to_numpy().astype(int)
    out["hourly_share_local"] = (np.bincount(hour, minlength=24) / len(hour)).round(4).tolist()
    return out


def daily_acf(tx: pd.DataFrame, n_days: int, lag: int = 7) -> float:
    """Lag-``lag`` autocorrelation of the daily legit-purchase count (weekly cycle)."""
    d = ((tx["event_time"] - tx["event_time"].min()).dt.total_seconds() // 86_400).astype(int)
    m = (tx["is_fraud_true"] == 0) & (tx["txn_type"] == "purchase") & (tx["recurring"] == 0)
    c = np.bincount(d[m], minlength=n_days)[:n_days].astype(float)
    c = c[7:-7] if len(c) > 30 else c
    c = c - c.mean()
    return float((c[:-lag] * c[lag:]).sum() / max((c * c).sum(), 1e-9))


PRESET_TARGETS = {
    "realistic": dict(rate=(0.0008, 0.0030), note="0.1-0.2% fraud attempts by count (assumption; card fraud ~0.06% by value, Nilson 2024)"),
    "realistic_dense": dict(rate=(0.0030, 0.0080), note="experiment world: prevalence deliberately ~3x realistic (not calibrated)"),
    "hard": dict(rate=(0.0010, 0.0040), note="like realistic, a bit more fraud"),
    "ieee_like": dict(rate=(0.027, 0.045), note="IEEE-CIS train 3.51%"),
}


def calibration_table(res: SimResult) -> pd.DataFrame:
    """Target statistic | real-world source | simulated value | tolerance | pass."""
    tx, cfg = res.transactions, res.metadata["config"]
    preset = cfg.get("preset", "realistic")
    tgt = PRESET_TARGETS.get(preset, PRESET_TARGETS["realistic"])
    lab = label_checks(res)
    v = _fraud_value_rates(tx)
    rows = []

    def add(stat, source, sim, lo, hi):
        ok = bool(lo <= sim <= hi) if sim == sim else False
        rows.append({"statistic": stat, "real_world_source": source, "simulated": round(float(sim), 5),
                     "tolerance": f"[{lo:g}, {hi:g}]", "pass": ok})

    add("fraud attempts / all rows (count)", tgt["note"], float(tx["is_fraud_true"].mean()), *tgt["rate"])
    if preset != "ieee_like":
        add("approved fraud value / approved card value", "Nilson 2024: 6.43c per $100 = 0.064% (tolerance ~x3)",
            v["fraud_value_rate"], 0.0002, 0.0070 if preset == "realistic_dense" else 0.0020)
        add("CNP share of card-fraud value", "UK Finance 2025 / ECB: 70-83% of fraud value is CNP",
            v["cnp_share_of_fraud_value"], 0.65, 0.88)
    cnp = tx[(tx["channel"] == "CNP") & (tx["txn_type"] == "purchase")]
    add("card-on-file share of CNP purchases", "~20% of CNP volume is stored credentials / recurring (task brief)",
        float((cnp["pos_entry_mode"] == "card_on_file").mean()), 0.08, 0.40)
    add("issuer+policy decline rate of legit purchases", "e-commerce approval ~90-97% (industry range)",
        float(1 - tx.loc[(tx["is_fraud_true"] == 0) & (tx["txn_type"] == "purchase"), "approved"].mean()), 0.015, 0.10)
    add("weekly seasonality: lag-7 autocorrelation of daily volume", "strong weekly cycle in retail payments",
        daily_acf(tx, cfg["n_days"]), 0.30 if cfg.get("calendar", True) else -1.0, 1.0)
    if cfg.get("label_mode") == "realistic":
        add("median chargeback delay (days)", "chargebacks arrive ~30-45 days after the purchase (Visa/MC dispute cycle)",
            lab["chargeback_delay_median_days"], 25, 50)
        add("max chargeback delay (days)", "120-day dispute window", lab["chargeback_delay_max_days"], 60, cfg["label_window_days"] + 1)
        add("friendly fraud share of disputes", "10-30% of fraud disputes are first-party / friendly",
            lab["friendly_share_of_disputes"], 0.10, 0.30)
        add("share of true fraud rows ever labelled fraud", "not all fraud is reported (assumption 35-90%)",
            lab["share_fraud_eventually_reported_1"], 0.35, 0.90)
        add("share of declined fraud with any outcome", "declines carry no outcome except contact-confirmed (cfg share)",
            lab["share_declined_fraud_with_outcome"], 0.0, 0.40)
    start = pd.Timestamp(res.metadata["start"])
    n_days = cfg["n_days"]
    if n_days >= 370 and cfg.get("calendar", True):
        day = ((tx["event_time"] - start).dt.total_seconds() // 86_400).astype(int)
        dt = start + pd.to_timedelta(np.arange(n_days), unit="D")
        cnt = np.bincount(day[tx["is_fraud_true"] == 0], minlength=n_days)[:n_days].astype(float)
        xmas = (dt.month == 12) & (dt.day >= 15) & (dt.day <= 24)
        if xmas.any():
            add("Dec 15-24 vs average-day volume", "Christmas run-up lifts retail volume ~+20-50%",
                float(cnt[xmas].mean() / cnt.mean()), 1.10, 1.80)
    return pd.DataFrame(rows)


def validate(res: SimResult, ieee_ref: dict | None = None, ceiling: bool = True) -> dict:
    val_start = pd.Timestamp(res.metadata["splits"]["val"][0])
    tx = res.transactions
    cal = calibration_table(res)
    pur = purity_check(tx)
    return {
        "summary": summary(res),
        "structure": structure_checks(res),
        "labels": label_checks(res),
        "calibration": cal.to_dict("records"),
        "calibration_pass_rate": float(cal["pass"].mean()) if len(cal) else float("nan"),
        "purity": {"violations": pur.to_dict("records"), "pass": bool(len(pur) == 0)},
        "row_only_ceiling": row_only_ceiling(tx, val_start) if ceiling else None,
        "leakage": leakage_scan(tx, val_start).to_dict("records"),
        "signals": scenario_signals(res).to_dict("records"),
        "ieee_reference": ieee_ref,
    }

"""
Public schema: turns internal entity *positions* into the columns a payments system would log.

Ids are random numbers assigned lazily (``IdBook``) so that an id never encodes creation order
(appended mule accounts, reissued cards and fraud devices would otherwise get the highest ids).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import geo
from .config import SimConfig
from .entities import World

# columns known BEFORE a decision (inputs to a decision policy)
PRE_COLUMNS = [
    "transaction_id", "event_time", "local_hour",
    "customer_id", "account_age_days", "email_domain", "billing_country", "address_id",
    "card_id", "prev_card_id", "card_bin", "card_country", "card_type", "card_network",
    "merchant_id", "mcc", "merchant_country", "merchant_age_days", "terminal_id",
    "channel", "pos_entry_mode", "txn_type", "recurring", "original_transaction_id",
    "amount", "currency", "amount_usd",
    "ip", "ip_country", "ip_lat", "ip_lon", "ip_type",
    "device_id", "device_os", "browser", "shipping_country",
    "counterparty_customer_id",
    "avs_result", "cvc_check", "three_ds_result",
]
DECISION_COLUMNS = ["decision", "risk_score", "propensity", "declined_by", "liability_shift",
                    "auth_response_code", "approved"]
LABEL_COLUMNS = ["reported_label", "label_status", "label_date", "chargeback_date", "dispute_reason"]
TRUTH_COLUMNS = ["is_fraud", "is_fraud_true", "fraud_scenario", "incident_id", "ring_id", "gang_id"]
COLUMN_ORDER = PRE_COLUMNS + DECISION_COLUMNS + LABEL_COLUMNS + TRUTH_COLUMNS

_RANGES = {"customer": (10_000_000, 999_999_999), "card": (10_000_000, 999_999_999),
           "device": (10_000_000, 999_999_999), "address": (10_000_000, 999_999_999),
           "merchant": (100_000, 9_999_999), "terminal": (100_000, 9_999_999)}


class IdBook:
    """Random unique public ids per entity kind, drawn lazily as entities appear."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.pool = {k: np.zeros(0, dtype=np.int64) for k in _RANGES}

    def ids(self, kind: str, n: int) -> np.ndarray:
        have = self.pool[kind]
        if len(have) < n:
            lo, hi = _RANGES[kind]
            need = n - len(have)
            while need > 0:
                x = np.unique(self.rng.integers(lo, hi, max(int(need * 1.3) + 64, 5000)))
                x = x[~np.isin(x, have)]
                self.rng.shuffle(x)
                have = np.concatenate([have, x])
                need = n - len(have)
            self.pool[kind] = have
        return self.pool[kind][:n]

    def map(self, kind: str, idx, size: int | None = None):
        idx = np.asarray(idx, dtype=np.int64)
        top = max(int(idx.max()) + 1, 1) if len(idx) else 1
        ids = self.ids(kind, max(top, size or 0))
        mask = idx < 0
        return pd.arrays.IntegerArray(ids[idx.clip(0)].astype(np.int64), mask)


class IpStr:
    """Incrementally cached dotted-quad strings of the IP table."""

    def __init__(self):
        self.arr = np.zeros(0, dtype=object)

    def get(self, world: World, idx: np.ndarray) -> np.ndarray:
        n = len(world.ips)
        if len(self.arr) < n:
            self.arr = np.concatenate([self.arr, geo.ip_to_str(world.ips["ip_int"][len(self.arr):n])])
        idx = np.asarray(idx, dtype=np.int64)
        out = self.arr[idx.clip(0)] if len(self.arr) else np.full(len(idx), None, dtype=object)
        out = out.copy()
        out[idx < 0] = None
        return out


def public_frame(core: pd.DataFrame, world: World, cfg: SimConfig, idb: IdBook, ipstr: IpStr,
                 avs, cvc, tds) -> pd.DataFrame:
    """Pre-decision public columns for one epoch (``core`` = internal rows sorted by t)."""
    C, K, M, D, I = world.customers, world.cards, world.merchants, world.devices, world.ips
    cty, cit, mccs = world.countries, world.cities, world.mccs
    code_of = cty["code"].to_numpy().astype(object)
    t = core["t"].to_numpy()
    cust = core["customer"].to_numpy()
    card = core["card"].to_numpy()
    merch = core["merchant"].to_numpy()
    dev = core["device"].to_numpy()
    ip = core["ip"].to_numpy()
    channel = core["channel"].to_numpy()
    guest = core["guest"].to_numpy() == 1
    start = pd.Timestamp(cfg.start_date, tz="UTC")
    event_time = start + pd.to_timedelta(t, unit="s")

    m_city = M["city"][merch]
    m_country = cit["country_idx"].to_numpy()[m_city]
    tz = np.where(channel == "CP", cit["tz"].to_numpy()[m_city],
                  np.where(ip >= 0, I["tz"][ip.clip(0)], np.where(cust >= 0, np.nan_to_num(C["tz"][cust.clip(0)]), 0.0)))
    local_hour = (((t + tz * 3600) % 86_400) // 3600).astype(np.int8)

    fx = cty["usd_per_unit"].to_numpy()[m_country]
    scale = 10.0 ** cty["decimals"].to_numpy()[m_country]
    value = core["value_usd"].to_numpy()
    amount = np.maximum(np.round(value / fx * scale), 1) / scale
    amount_usd = np.round(amount * fx, 4)

    mcc_idx = M["mcc_idx"][merch]
    mcc_code = np.where(mcc_idx >= 0, mccs["mcc"].to_numpy().astype(object)[mcc_idx.clip(0)], geo.P2P_MCC)
    has_acct = (cust >= 0) & ~guest & (C["kind"][cust.clip(0)] != "phantom")
    age = np.floor((t - np.where(cust >= 0, C["open_t"][cust.clip(0)], np.nan)) / 86_400)
    age = np.where(has_acct, age, np.nan)
    m_age = np.floor((t - M["open_t"][merch]) / 86_400)
    m_age = np.where(np.isfinite(m_age), np.maximum(m_age, 0), np.nan)
    ship = core["shipping_country"].to_numpy()
    uid = core["uid"].to_numpy()
    ref = core["refund_of"].to_numpy()
    show_cust = (cust >= 0) & ~guest
    ip_s = ipstr.get(world, ip)

    def lk(idx, values):
        idx = np.asarray(idx, dtype=np.int64)
        out = np.asarray(values, dtype=object)[idx.clip(0)].copy()
        out[idx < 0] = None
        return out

    return pd.DataFrame({
        "transaction_id": uid + 1,
        "event_time": event_time,
        "local_hour": local_hour,
        "customer_id": idb.map("customer", np.where(show_cust, cust, -1)),
        "account_age_days": pd.Series(age).astype("Int64").array,
        "email_domain": core["email_domain"].to_numpy(),
        "billing_country": np.where(cust >= 0, code_of[C["country"][cust.clip(0)]], code_of[K["country"][card]]),
        "address_id": idb.map("address", np.where(cust >= 0, C["address"][cust.clip(0)], -1)),
        "card_id": idb.map("card", card),
        "prev_card_id": idb.map("card", K["prev"][card]),
        "card_bin": K["bin"][card].astype(np.int64),
        "card_country": code_of[K["country"][card]],
        "card_type": K["card_type"][card],
        "card_network": K["network"][card],
        "merchant_id": idb.map("merchant", merch),
        "mcc": mcc_code,
        "merchant_country": code_of[m_country],
        "merchant_age_days": pd.Series(m_age).astype("Int64").array,
        "terminal_id": idb.map("terminal", core["terminal"].to_numpy()),
        "channel": channel,
        "pos_entry_mode": core["pos_entry_mode"].to_numpy(),
        "txn_type": core["txn_type"].to_numpy(),
        "recurring": core["recurring"].to_numpy().astype(np.int8),
        "original_transaction_id": pd.arrays.IntegerArray(np.where(ref >= 0, ref + 1, 0).astype(np.int64), ref < 0),
        "amount": amount,
        "currency": cty["currency"].to_numpy().astype(object)[m_country],
        "amount_usd": amount_usd,
        "ip": ip_s,
        "ip_country": np.where(ip >= 0, code_of[I["country"][ip.clip(0)]], None).astype(object),
        "ip_lat": np.where(ip >= 0, np.round(I["lat"][ip.clip(0)], 4), np.nan),
        "ip_lon": np.where(ip >= 0, np.round(I["lon"][ip.clip(0)], 4), np.nan),
        "ip_type": np.where(ip >= 0, I["ip_type"][ip.clip(0)], None).astype(object),
        "device_id": idb.map("device", dev),
        "device_os": np.where(dev >= 0, D["os"][dev.clip(0)], None).astype(object),
        "browser": np.where(dev >= 0, D["browser"][dev.clip(0)], None).astype(object),
        "shipping_country": lk(ship, code_of),
        "counterparty_customer_id": idb.map("customer", core["counterparty"].to_numpy()),
        "avs_result": avs, "cvc_check": cvc, "three_ds_result": tds,
    })

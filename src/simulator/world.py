"""
Orchestration: entities -> legit behaviour -> fraud agents -> authorisation ->
friendly-fraud disputes -> label delays -> public schema.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from . import geo
from .auth import apply_auth
from .config import SimConfig
from .entities import World, build_world
from .legit import generate_legit
from .rows import Rows
from .scenarios import FraudFactory

GENERATOR_VERSION = "sim-v1"

# Columns that must never be used as model features.
GROUND_TRUTH_COLS = ["is_fraud", "label_date", "chargeback_date", "fraud_scenario", "incident_id", "ring_id"]
# Outcome of the authorisation decision: known only AFTER scoring. Usable only from
# PRIOR rows (e.g. decline velocity), never as a feature of the current row.
OUTCOME_COLS = ["auth_response_code", "approved"]
ID_COLS = ["transaction_id", "customer_id", "card_id", "merchant_id", "terminal_id", "device_id",
           "address_id", "counterparty_customer_id"]


@dataclass
class SimResult:
    transactions: pd.DataFrame
    entities: Dict[str, pd.DataFrame]
    ground_truth: Dict[str, pd.DataFrame]
    metadata: dict = field(default_factory=dict)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        (out / "entities").mkdir(parents=True, exist_ok=True)
        (out / "ground_truth").mkdir(parents=True, exist_ok=True)
        self.transactions.to_parquet(out / "transactions.parquet", index=False)
        for name, df in self.entities.items():
            df.to_parquet(out / "entities" / f"{name}.parquet", index=False)
        for name, df in self.ground_truth.items():
            df.to_parquet(out / "ground_truth" / f"{name}.parquet", index=False)
        (out / "metadata.json").write_text(json.dumps(self.metadata, indent=2, default=str), encoding="utf-8")
        return out


def _random_ids(rng, n: int, lo: int = 10_000_000, hi: int = 999_999_999) -> np.ndarray:
    x = np.unique(rng.integers(lo, hi, int(n * 1.2) + 16))
    while len(x) < n:
        x = np.unique(np.concatenate([x, rng.integers(lo, hi, n)]))
    rng.shuffle(x)
    return x[:n].astype(np.int64)


def _map(idx: np.ndarray, ids: np.ndarray) -> pd.arrays.IntegerArray:
    idx = np.asarray(idx, dtype=np.int64)
    mask = idx < 0
    return pd.arrays.IntegerArray(ids[idx.clip(0)].astype(np.int64), mask)


def _lookup(idx: np.ndarray, values: np.ndarray, missing=None) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    out = np.asarray(values, dtype=object)[idx.clip(0)]
    out[idx < 0] = missing
    return out


def simulate(cfg: SimConfig | None = None, verbose: bool = False) -> SimResult:
    cfg = cfg or SimConfig()
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    log = print if verbose else (lambda *a, **k: None)

    world = build_world(cfg, rng)
    log(f"[sim] world: {cfg.n_customers:,} customers, {len(world.cards):,} cards, "
        f"{len(world.merchants):,} merchants, {len(world.terminals):,} terminals, {len(world.trips):,} trips")
    legit = generate_legit(world, rng)
    log(f"[sim] legit rows: {len(legit['t']):,}")
    ff = FraudFactory(world, legit, rng)
    ff.run(len(legit["t"]))
    log(f"[sim] fraud rows (pre-auth): {sum(len(p['t']) for p in ff.rows.parts):,}  "
        f"incidents: {len(ff.incidents):,}  rings: {len(ff.rings)}")

    base = Rows()
    base.add_frame(legit)
    core = pd.concat([base.to_frame(), ff.rows.to_frame()], ignore_index=True)
    T = cfg.horizon_sec
    if ff.bust_out_end:
        end = core["customer"].map(ff.bust_out_end).to_numpy(dtype=float)
        abandoned = (core["is_fraud"].to_numpy() == 0) & ~np.isnan(end) & (core["t"].to_numpy() > end)
        core = core[~abandoned]
    core = core[(core["t"] >= 0) & (core["t"] < T)]
    core = core.sort_values("t", kind="stable").reset_index(drop=True)

    avs, cvc, tds, code = apply_auth(core, world, rng)
    approved = code == "00"

    # friendly fraud: approved legit CNP purchases later disputed by the cardholder
    cand = np.nonzero((core["is_fraud"].to_numpy() == 0) & (core["channel"].to_numpy() == "CNP")
                      & approved & (core["value_usd"].to_numpy() > 10))[0]
    k = min(ff.friendly_budget, len(cand))
    pick = rng.choice(cand, k, replace=False)
    core.loc[pick, "is_fraud"] = 1
    core.loc[pick, "scenario"] = "friendly_fraud"
    core.loc[pick, "delay_days"] = rng.uniform(14, 90, k)
    # declined fraud attempts are confirmed quickly (cardholder contacted / attack reviewed)
    df_mask = (core["is_fraud"].to_numpy() == 1) & ~approved
    core.loc[df_mask, "delay_days"] = rng.uniform(0.1, 3, df_mask.sum())

    tx = _assemble(core, world, cfg, rng, avs, cvc, tds, code)
    # incidents / rings whose rows all fell outside the simulated window are not real here
    present = set(np.unique(core["incident"].to_numpy()).tolist())
    ff.incidents = [i for i in ff.incidents if i["incident"] in present]
    present_rings = set(np.unique(core["ring"].to_numpy()).tolist())
    ff.rings = [r for r in ff.rings if r["ring"] in present_rings]
    ff.ring_members = [m for m in ff.ring_members if m["ring"] in present_rings]
    entities, ground_truth, id_maps = _entities(world, ff, rng)
    # id maps are drawn inside _entities; re-map transaction ids with the same maps
    for col, (src, key) in {"customer_id": ("customer", "customer"), "card_id": ("card", "card"),
                            "merchant_id": ("merchant", "merchant"), "terminal_id": ("terminal", "terminal"),
                            "device_id": ("device", "device"), "counterparty_customer_id": ("counterparty", "customer"),
                            "address_id": ("_address", "address")}.items():
        idx = core[src].to_numpy() if src != "_address" else np.where(
            core["customer"].to_numpy() >= 0, world.customers["address"][core["customer"].to_numpy().clip(0)], -1)
        tx[col] = _map(idx, id_maps[key])
    tx = tx[_COLUMN_ORDER]

    meta = {
        "generator": GENERATOR_VERSION,
        "config": cfg.to_dict(),
        "start": str(pd.Timestamp(cfg.start_date, tz="UTC")),
        "splits": {
            "train": [str(pd.Timestamp(cfg.start_date, tz="UTC")),
                      str(pd.Timestamp(cfg.start_date, tz="UTC") + pd.Timedelta(seconds=cfg.val_start_sec))],
            "val": [str(pd.Timestamp(cfg.start_date, tz="UTC") + pd.Timedelta(seconds=cfg.val_start_sec)),
                    str(pd.Timestamp(cfg.start_date, tz="UTC") + pd.Timedelta(seconds=cfg.test_start_sec))],
            "test": [str(pd.Timestamp(cfg.start_date, tz="UTC") + pd.Timedelta(seconds=cfg.test_start_sec)),
                     str(pd.Timestamp(cfg.start_date, tz="UTC") + pd.Timedelta(seconds=T))],
        },
        "ground_truth_columns": GROUND_TRUTH_COLS,
        "outcome_columns": OUTCOME_COLS,
        "id_columns": ID_COLS,
        "novel_scenarios": ["bust_out"],
        "n_rows": int(len(tx)),
        "fraud_rate": float(tx["is_fraud"].mean()),
        "fraud_budgets": ff.budgets | {"friendly_fraud": ff.friendly_budget},
    }
    return SimResult(tx, entities, ground_truth, meta)


_COLUMN_ORDER = [
    "transaction_id", "event_time", "local_hour",
    "customer_id", "account_age_days", "email_domain", "billing_country", "address_id",
    "card_id", "card_bin", "card_country", "card_type", "card_network",
    "merchant_id", "mcc", "merchant_country", "terminal_id",
    "channel", "pos_entry_mode", "amount", "currency", "amount_usd",
    "ip", "ip_country", "ip_lat", "ip_lon", "ip_type",
    "device_id", "device_os", "browser", "shipping_country",
    "counterparty_customer_id",
    "avs_result", "cvc_check", "three_ds_result",
    "auth_response_code", "approved",
    "is_fraud", "label_date", "chargeback_date", "fraud_scenario", "incident_id", "ring_id",
]


def _assemble(core, world: World, cfg: SimConfig, rng, avs, cvc, tds, code) -> pd.DataFrame:
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
    start = pd.Timestamp(cfg.start_date, tz="UTC")
    event_time = start + pd.to_timedelta(t, unit="s")

    m_city = M["city"][merch]
    m_country = cit["country_idx"].to_numpy()[m_city]
    tz = np.where(channel == "CP", cit["tz"].to_numpy()[m_city],
                  np.where(ip >= 0, I["tz"][ip.clip(0)],
                           np.where(cust >= 0, C["tz"][cust.clip(0)], 0.0)))
    local_hour = (((t + tz * 3600) % 86_400) // 3600).astype(np.int8)

    fx = cty["usd_per_unit"].to_numpy()[m_country]
    scale = 10.0 ** cty["decimals"].to_numpy()[m_country]
    value = core["value_usd"].to_numpy()
    amount = np.maximum(np.round(value / fx * scale), 1) / scale
    amount_usd = np.round(amount * fx, 4)

    ip_str = geo.ip_to_str(I["ip_int"])
    mcc_idx = M["mcc_idx"][merch]
    mcc_code = np.where(mcc_idx >= 0, mccs["mcc"].to_numpy().astype(object)[mcc_idx.clip(0)], geo.P2P_MCC)
    open_t = np.where(cust >= 0, C["open_t"][cust.clip(0)], np.nan)
    age = np.floor((t - open_t) / 86_400)
    ship = core["shipping_country"].to_numpy()
    is_fraud = core["is_fraud"].to_numpy().astype(np.int8)
    approved = (code == "00")
    delay = pd.to_timedelta(core["delay_days"].to_numpy() * 86_400, unit="s")
    label_date = pd.Series(event_time + delay).where(is_fraud == 1)
    chargeback = label_date.where(approved & (channel != "P2P"))

    def _ids(x):
        x = np.asarray(x, dtype=np.int64)
        return pd.arrays.IntegerArray(np.where(x >= 0, x, 0), x < 0)

    return pd.DataFrame({
        "transaction_id": np.arange(1, len(core) + 1, dtype=np.int64),
        "event_time": event_time,
        "local_hour": local_hour,
        "account_age_days": pd.Series(age).astype("Int64").array,
        "email_domain": core["email_domain"].to_numpy(),
        "billing_country": np.where(cust >= 0, code_of[C["country"][cust.clip(0)]], code_of[K["country"][card]]),
        "card_bin": K["bin"][card].astype(np.int64),
        "card_country": code_of[K["country"][card]],
        "card_type": K["card_type"][card],
        "card_network": K["network"][card],
        "mcc": mcc_code,
        "merchant_country": code_of[m_country],
        "channel": channel,
        "pos_entry_mode": core["pos_entry_mode"].to_numpy(),
        "amount": amount,
        "currency": cty["currency"].to_numpy().astype(object)[m_country],
        "amount_usd": amount_usd,
        "ip": _lookup(ip, ip_str),
        "ip_country": _lookup(ip, code_of[I["country"]]),
        "ip_lat": np.where(ip >= 0, np.round(I["lat"][ip.clip(0)], 4), np.nan),
        "ip_lon": np.where(ip >= 0, np.round(I["lon"][ip.clip(0)], 4), np.nan),
        "ip_type": _lookup(ip, I["ip_type"]),
        "device_os": _lookup(dev, D["os"]),
        "browser": _lookup(dev, D["browser"]),
        "shipping_country": _lookup(ship, code_of),
        "avs_result": avs,
        "cvc_check": cvc,
        "three_ds_result": tds,
        "auth_response_code": code,
        "approved": approved.astype(np.int8),
        "is_fraud": is_fraud,
        "label_date": label_date.to_numpy(),
        "chargeback_date": chargeback.to_numpy(),
        "fraud_scenario": core["scenario"].to_numpy(),
        "incident_id": _ids(core["incident"].to_numpy()),
        "ring_id": _ids(core["ring"].to_numpy()),
    })


def _entities(world: World, ff: FraudFactory, rng):
    C, K, M, TT, D, I = world.customers, world.cards, world.merchants, world.terminals, world.devices, world.ips
    cty, cit, mccs = world.countries, world.cities, world.mccs
    code_of = cty["code"].to_numpy().astype(object)
    ids = {
        "customer": _random_ids(rng, len(C)),
        "card": _random_ids(rng, len(K)),
        "merchant": _random_ids(rng, len(M), 100_000, 9_999_999),
        "terminal": _random_ids(rng, len(TT), 100_000, 9_999_999),
        "device": _random_ids(rng, len(D)),
        "address": _random_ids(rng, int(C["address"].max()) + 1),
    }
    start = pd.Timestamp(world.cfg.start_date, tz="UTC")
    mcc_idx = M["mcc_idx"]
    customers = pd.DataFrame({
        "customer_id": ids["customer"],
        "home_country": code_of[C["country"]],
        "home_city": cit["city"].to_numpy()[C["city"]],
        "account_open_date": start + pd.to_timedelta(C["open_t"], unit="s"),
        "email_domain": C["email"],
        "address_id": ids["address"][C["address"]],
        "n_cards": C["n_cards"],
    })
    cards = pd.DataFrame({
        "card_id": ids["card"], "customer_id": _map(K["owner"], ids["customer"]),
        "card_bin": K["bin"].astype(np.int64), "card_country": code_of[K["country"]],
        "card_network": K["network"], "card_type": K["card_type"],
    })
    merchants = pd.DataFrame({
        "merchant_id": ids["merchant"],
        "mcc": np.where(mcc_idx >= 0, mccs["mcc"].to_numpy().astype(object)[mcc_idx.clip(0)], geo.P2P_MCC),
        "mcc_name": np.where(mcc_idx >= 0, mccs["name"].to_numpy().astype(object)[mcc_idx.clip(0)], "p2p_platform"),
        "merchant_country": code_of[cit["country_idx"].to_numpy()[M["city"]]],
        "city": cit["city"].to_numpy()[M["city"]], "sales_mode": M["mode"], "uses_3ds": M["uses_3ds"],
    })
    terminals = pd.DataFrame({"terminal_id": ids["terminal"], "merchant_id": ids["merchant"][TT["merchant"]],
                              "virtual": TT["virtual"]})
    devices = pd.DataFrame({"device_id": ids["device"], "device_os": D["os"], "browser": D["browser"]})
    ips = pd.DataFrame({"ip": geo.ip_to_str(I["ip_int"]), "ip_country": code_of[I["country"]],
                        "ip_city": cit["city"].to_numpy()[I["city"]], "ip_lat": I["lat"], "ip_lon": I["lon"],
                        "ip_type": I["ip_type"]}).drop_duplicates("ip")

    # ── ground truth ──
    def _ts(x):
        return start + pd.to_timedelta(np.asarray(x, dtype=float), unit="s")

    inc = pd.DataFrame(ff.incidents)
    if len(inc):
        for col in ("start", "end", "skim_start", "skim_end"):
            if col in inc:
                inc[col] = _ts(inc[col].to_numpy(dtype=float))
        for col, key in (("customer", "customer"), ("card", "card"), ("merchant", "merchant"), ("terminal", "terminal")):
            if col in inc:
                inc[col + "_id"] = _map(inc[col].fillna(-1).to_numpy(dtype=np.int64), ids[key])
                inc = inc.drop(columns=col)
        inc = inc.rename(columns={"incident": "incident_id", "ring": "ring_id", "gang": "gang_id"})
        for col in ("ring_id", "gang_id"):
            if col in inc:
                inc[col] = inc[col].astype("Int64")
    rings = pd.DataFrame([{k: v for k, v in r.items() if k != "entries"} for r in ff.rings])
    if len(rings):
        rings["start"] = _ts(rings["start"])
        rings["end"] = _ts(rings["end"])
        rings["city"] = cit["city"].to_numpy()[rings["city"].to_numpy()]
        rings = rings.rename(columns={"ring": "ring_id", "incident": "incident_id"})
    members = pd.DataFrame(ff.ring_members)
    if len(members):
        members["customer_id"] = ids["customer"][members["customer"].to_numpy()]
        members = members.drop(columns="customer").rename(columns={"ring": "ring_id"})
    gang_rows = []
    for g, gang in enumerate(ff.gangs):
        gang_rows += [{"gang_id": g, "kind": "device", "value": str(ids["device"][d])} for d in gang["devices"]]
        for kind, arr in gang["ips"].items():
            ip_s = geo.ip_to_str(I["ip_int"][arr])
            gang_rows += [{"gang_id": g, "kind": f"ip_{kind}", "value": s} for s in ip_s]
    customer_truth = pd.DataFrame({"customer_id": ids["customer"], "is_mule": C["is_mule"],
                                   "account_kind": C["kind"]})
    entities = {"customers": customers, "cards": cards, "merchants": merchants, "terminals": terminals,
                "devices": devices, "ips": ips}
    ground_truth = {"incidents": inc, "rings": rings, "ring_members": members,
                    "gang_infrastructure": pd.DataFrame(gang_rows), "customers": customer_truth}
    return entities, ground_truth, ids

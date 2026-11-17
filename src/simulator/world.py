"""
Orchestration: entities -> legit behaviour -> fraud agents -> closed loop (policy decisions,
authorisation, labels, card reissue, attacker adaptation) -> account events -> public schema.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from . import geo
from .assemble import (COLUMN_ORDER, DECISION_COLUMNS, LABEL_COLUMNS, TRUTH_COLUMNS, IdBook, IpStr)
from .config import NOVEL_SCENARIO, SimConfig
from .entities import World, build_world
from .events import build_events
from .legit import generate_legit
from .loop import ClosedLoop, DecisionPolicy, RulesPolicy
from .scenarios import FraudFactory

GENERATOR_VERSION = "sim-v2"

# Columns that must never be used as model features (they are produced by the fraud process itself).
GROUND_TRUTH_COLS = list(TRUTH_COLUMNS)
# Labels as a real system would see them: usable only from rows whose ``label_date`` is in the past.
LABEL_COLS = list(LABEL_COLUMNS)
# Outcome of the decision: known only AFTER scoring. Usable only from PRIOR rows
# (e.g. decline velocity), never as a feature of the current row.
OUTCOME_COLS = list(DECISION_COLUMNS)
ID_COLS = ["transaction_id", "customer_id", "card_id", "prev_card_id", "merchant_id", "terminal_id", "device_id",
           "address_id", "counterparty_customer_id", "original_transaction_id"]
_COLUMN_ORDER = COLUMN_ORDER


@dataclass
class SimResult:
    transactions: pd.DataFrame
    entities: Dict[str, pd.DataFrame]
    ground_truth: Dict[str, pd.DataFrame]
    metadata: dict = field(default_factory=dict)
    events: pd.DataFrame | None = None

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        (out / "entities").mkdir(parents=True, exist_ok=True)
        (out / "ground_truth").mkdir(parents=True, exist_ok=True)
        self.transactions.to_parquet(out / "transactions.parquet", index=False)
        if self.events is not None:
            self.events.to_parquet(out / "events.parquet", index=False)
        for name, df in self.entities.items():
            df.to_parquet(out / "entities" / f"{name}.parquet", index=False)
        for name, df in self.ground_truth.items():
            df.to_parquet(out / "ground_truth" / f"{name}.parquet", index=False)
        (out / "metadata.json").write_text(json.dumps(self.metadata, indent=2, default=str), encoding="utf-8")
        return out


def simulate(cfg: SimConfig | None = None, verbose: bool = False, policy: DecisionPolicy | None = None) -> SimResult:
    """Generate a world. ``policy`` defaults to the rules engine (``loop.RulesPolicy``)."""
    cfg = cfg or SimConfig()
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    log = print if verbose else (lambda *a, **k: None)
    policy = policy or RulesPolicy()

    world = build_world(cfg, rng)
    log(f"[sim] world: {cfg.n_customers:,} customers, {len(world.cards):,} cards, "
        f"{len(world.merchants):,} merchants, {len(world.terminals):,} terminals, {len(world.trips):,} trips")
    legit = generate_legit(world, rng)
    log(f"[sim] legit rows: {len(legit['t']):,}")
    ff = FraudFactory(world, legit, rng)
    idb, ipstr = IdBook(rng), IpStr()
    loop = ClosedLoop(world, legit, ff, policy, rng, idb, ipstr)
    tx = loop.run(verbose=verbose)
    tx = tx[COLUMN_ORDER].reset_index(drop=True)
    pos = {k: np.concatenate([p[k] for p in loop.pos]) for k in loop.pos[0]}
    log(f"[sim] rows: {len(tx):,}  fraud rate {tx['is_fraud'].mean():.3%}  incidents: {len(ff.incidents):,}")

    events = build_events(tx, pos, world, ff, idb, ipstr, cfg, rng)
    log(f"[sim] account events: {len(events):,}")

    present = set(tx["incident_id"].dropna().astype(int).tolist())
    ff.incidents = [i for i in ff.incidents if i["incident"] in present]
    present_rings = set(tx["ring_id"].dropna().astype(int).tolist())
    ff.rings = [r for r in ff.rings if r["ring"] in present_rings]
    ff.ring_members = [m for m in ff.ring_members if m["ring"] in present_rings]
    entities, ground_truth = _entities(world, ff, idb, ipstr, loop, cfg)

    T = cfg.horizon_sec
    t0 = pd.Timestamp(cfg.start_date, tz="UTC")
    meta = {
        "generator": GENERATOR_VERSION,
        "config": cfg.to_dict(),
        "policy": getattr(policy, "name", type(policy).__name__),
        "start": str(t0),
        "splits": {
            "train": [str(t0), str(t0 + pd.Timedelta(seconds=cfg.val_start_sec))],
            "val": [str(t0 + pd.Timedelta(seconds=cfg.val_start_sec)), str(t0 + pd.Timedelta(seconds=cfg.test_start_sec))],
            "test": [str(t0 + pd.Timedelta(seconds=cfg.test_start_sec)), str(t0 + pd.Timedelta(seconds=T))],
        },
        "ground_truth_columns": GROUND_TRUTH_COLS,
        "label_columns": LABEL_COLS,
        "outcome_columns": OUTCOME_COLS,
        "id_columns": ID_COLS,
        "novel_scenarios": [NOVEL_SCENARIO],
        "n_rows": int(len(tx)),
        "n_events": int(len(events)),
        "fraud_rate": float(tx["is_fraud_true"].mean()),
        "fraud_budgets": ff.budgets,
        "loop_stats": loop.stats,
        "label_window_days": cfg.label_window_days,
    }
    return SimResult(tx, entities, ground_truth, meta, events)


def _entities(world: World, ff: FraudFactory, idb: IdBook, ipstr: IpStr, loop: ClosedLoop, cfg: SimConfig):
    C, K, M, TT, D, I = world.customers, world.cards, world.merchants, world.terminals, world.devices, world.ips
    cty, cit, mccs = world.countries, world.cities, world.mccs
    code_of = cty["code"].to_numpy().astype(object)
    start = pd.Timestamp(cfg.start_date, tz="UTC")
    ids = {"customer": idb.ids("customer", len(C)), "card": idb.ids("card", len(K)),
           "merchant": idb.ids("merchant", len(M)), "terminal": idb.ids("terminal", len(TT)),
           "device": idb.ids("device", len(D)), "address": idb.ids("address", int(C["address"].max()) + 1)}
    mcc_idx = M["mcc_idx"]

    def ts(x):
        return start + pd.to_timedelta(np.asarray(x, dtype=float), unit="s")

    phantom = C["kind"] == "phantom"
    customers = pd.DataFrame({
        "customer_id": ids["customer"],
        "home_country": code_of[C["country"]],
        "home_city": cit["city"].to_numpy()[C["city"]],
        "account_open_date": ts(np.where(phantom, np.nan, C["open_t"])),
        "email_domain": C["email"],
        "address_id": idb.map("address", C["address"]),
        "n_cards": C["n_cards"],
    })
    cards = pd.DataFrame({
        "card_id": ids["card"], "customer_id": idb.map("customer", K["owner"]),
        "prev_card_id": idb.map("card", K["prev"]),
        "card_bin": K["bin"].astype(np.int64), "card_country": code_of[K["country"]],
        "card_network": K["network"], "card_type": K["card_type"],
        "issued_date": ts(np.where(K["issued_t"] < -1e14, np.nan, K["issued_t"])),
        "blocked_date": ts(K["blocked_t"]),
    })
    merchants = pd.DataFrame({
        "merchant_id": ids["merchant"],
        "mcc": np.where(mcc_idx >= 0, mccs["mcc"].to_numpy().astype(object)[mcc_idx.clip(0)], geo.P2P_MCC),
        "mcc_name": np.where(mcc_idx >= 0, mccs["name"].to_numpy().astype(object)[mcc_idx.clip(0)], "p2p_platform"),
        "merchant_country": code_of[cit["country_idx"].to_numpy()[M["city"]]],
        "city": cit["city"].to_numpy()[M["city"]], "sales_mode": M["mode"], "uses_3ds": M["uses_3ds"],
        "open_date": ts(M["open_t"]),
    })
    terminals = pd.DataFrame({"terminal_id": ids["terminal"], "merchant_id": ids["merchant"][TT["merchant"]],
                              "virtual": TT["virtual"]})
    devices = pd.DataFrame({"device_id": ids["device"], "device_os": D["os"], "browser": D["browser"]})
    ips = pd.DataFrame({"ip": ipstr.get(world, np.arange(len(I))), "ip_country": code_of[I["country"]],
                        "ip_city": cit["city"].to_numpy()[I["city"]], "ip_lat": I["lat"], "ip_lon": I["lon"],
                        "ip_type": I["ip_type"]}).drop_duplicates("ip")

    inc = pd.DataFrame(ff.incidents)
    if len(inc):
        for col in ("start", "end", "skim_start", "skim_end", "spree_start"):
            if col in inc:
                inc[col] = ts(inc[col].to_numpy(dtype=float))
        for col, key in (("customer", "customer"), ("card", "card"), ("merchant", "merchant"), ("terminal", "terminal")):
            if col in inc:
                inc[col + "_id"] = idb.map(key, inc[col].fillna(-1).to_numpy(dtype=np.int64))
                inc = inc.drop(columns=col)
        inc = inc.rename(columns={"incident": "incident_id", "ring": "ring_id", "gang": "gang_id"})
        for col in ("ring_id", "gang_id"):
            if col in inc:
                inc[col] = inc[col].astype("Int64")
    rings = pd.DataFrame([{k: v for k, v in r.items() if k != "entries"} for r in ff.rings])
    if len(rings):
        rings["start"] = ts(rings["start"])
        rings["end"] = ts(rings["end"])
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
            gang_rows += [{"gang_id": g, "kind": f"ip_{kind}", "value": s} for s in ipstr.get(world, arr)]
    customer_truth = pd.DataFrame({"customer_id": ids["customer"], "is_mule": C["is_mule"], "account_kind": C["kind"]})
    merchant_truth = pd.DataFrame({"merchant_id": ids["merchant"], "merchant_kind": M["kind"]})
    # tactic statistics of the gangs at the end of the run (what they believe works)
    tac = ff.tactics
    tactics = pd.DataFrame([{"gang_id": g, "dim": d, "arm": a, "approvals": float(tac.succ[g, i, a]),
                             "attempts": float(tac.n[g, i, a])}
                            for g in range(tac.succ.shape[0]) for d, i in tac.DIMS.items() for a in range(3)])
    entities = {"customers": customers, "cards": cards, "merchants": merchants, "terminals": terminals,
                "devices": devices, "ips": ips}
    ground_truth = {"incidents": inc, "rings": rings, "ring_members": members,
                    "gang_infrastructure": pd.DataFrame(gang_rows), "customers": customer_truth,
                    "merchants": merchant_truth, "gang_tactics": tactics}
    return entities, ground_truth

"""
Account-event stream (``events.parquet``): login, password_reset, email_change, phone_change,
new_payee, device_enrolment - for legit customers AND takeover attackers.

Legit users also enrol new devices, reset passwords, change email addresses and log in from new
IPs, so the signal is real but not pure. An account takeover looks like:
failed logins -> login from a NEW device -> credential change -> dwell -> cash-out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .assemble import IdBook, IpStr
from .config import SimConfig
from .entities import World
from .legit import current_device, home_ip, mobile_ip

DAY = 86_400
EVENT_COLUMNS = ["event_id", "event_time", "customer_id", "event_type", "success", "device_id", "ip",
                 "ip_country", "ip_type", "device_os", "browser", "email_domain", "counterparty_customer_id"]
EVENT_TRUTH_COLUMNS = ["is_attacker", "incident_id"]
# scenarios whose purchases are made by the account owner's own session
OWNER_SESSION = {"legit", "friendly_fraud", "synthetic_identity_build", "synthetic_identity", "bust_out",
                 "mule_ring", "merchant_collusion"}


def _cols(world: World, idb: IdBook, ipstr: IpStr, cust, dev, ip) -> dict:
    D, I = world.devices, world.ips
    code_of = world.countries["code"].to_numpy().astype(object)
    dev, ip = np.asarray(dev, np.int64), np.asarray(ip, np.int64)
    return {"customer_id": idb.map("customer", cust), "device_id": idb.map("device", dev),
            "ip": ipstr.get(world, ip),
            "ip_country": np.where(ip >= 0, code_of[I["country"][ip.clip(0)]], None).astype(object),
            "ip_type": np.where(ip >= 0, I["ip_type"][ip.clip(0)], None).astype(object),
            "device_os": np.where(dev >= 0, D["os"][dev.clip(0)], None).astype(object),
            "browser": np.where(dev >= 0, D["browser"][dev.clip(0)], None).astype(object)}


def build_events(tx: pd.DataFrame, pos: dict, world: World, ff, idb: IdBook, ipstr: IpStr, cfg: SimConfig,
                 rng: np.random.Generator) -> pd.DataFrame:
    C = world.customers
    nc, T = cfg.n_customers, cfg.horizon_sec
    start = pd.Timestamp(cfg.start_date, tz="UTC")
    frames = []

    def add(t, cust, dev, ip, etype, success=True, email=None, payee=-1, attacker=0, incident=-1):
        n = len(t)
        if n == 0:
            return
        d = _cols(world, idb, ipstr, cust, dev, ip)
        d["event_time"] = start + pd.to_timedelta(np.asarray(t, float), unit="s")
        d["event_type"] = np.full(n, etype, dtype=object)
        d["success"] = np.broadcast_to(np.asarray(success, bool), (n,)).copy()
        d["email_domain"] = np.broadcast_to(np.asarray(email, dtype=object), (n,)).copy() if email is not None else np.full(n, None, dtype=object)
        d["counterparty_customer_id"] = idb.map("customer", np.broadcast_to(np.asarray(payee, np.int64), (n,)))
        d["is_attacker"] = np.broadcast_to(np.asarray(attacker, np.int8), (n,)).copy()
        d["incident_id"] = pd.arrays.IntegerArray(np.where(np.asarray(incident) >= 0, incident, 0).astype(np.int64) * np.ones(n, np.int64),
                                                  np.broadcast_to(np.asarray(incident) < 0, (n,)).copy())
        frames.append(pd.DataFrame(d))

    # (a) logins that precede online purchases by the account owner
    scen = tx["fraud_scenario"].to_numpy()
    pos_mode = tx["pos_entry_mode"].to_numpy()
    sel = (np.isin(scen, list(OWNER_SESSION)) & np.isin(pos_mode, ["ecom", "p2p"]) & (pos["customer"] >= 0)
           & (pos["guest"] == 0) & (pos["device"] >= 0) & (tx["txn_type"].to_numpy() == "purchase"))
    sel &= rng.random(len(tx)) < 0.55
    i = np.nonzero(sel)[0]
    t_sec = pos["t"][i] - rng.uniform(20, 900, len(i))
    ok = t_sec >= 0
    i, t_sec = i[ok], t_sec[ok]
    fail = rng.random(len(i)) < 0.025
    add(t_sec[~fail], pos["customer"][i[~fail]], pos["device"][i[~fail]], pos["ip"][i[~fail]], "login")
    add(t_sec[fail], pos["customer"][i[fail]], pos["device"][i[fail]], pos["ip"][i[fail]], "login", success=False)
    # (b) background logins (app opens, balance checks) without a purchase
    open_t = np.maximum(C["open_t"][:nc], 0)
    n_bg = rng.poisson(0.03 * C["online"][:nc] * (T - open_t) / DAY)
    cust = np.repeat(np.arange(nc), n_bg)
    tt = open_t[cust] + rng.random(len(cust)) * (T - open_t[cust])
    tt = tt - (tt % DAY) + ((C["peak_hour"][cust] + rng.normal(0, 1, len(cust)) * C["hour_spread"][cust]) % 24) * 3600 \
        - C["tz"][cust] * 3600
    ok = (tt >= open_t[cust]) & (tt < T)
    cust, tt = cust[ok], tt[ok]
    dev = current_device(rng, world, cust, tt)
    u = rng.random(len(cust))
    ip = np.where(u < 0.7, home_ip(world, cust, tt), mobile_ip(rng, world, C["city"][cust]))
    ok_login = rng.random(len(cust)) > 0.02
    add(tt[ok_login], cust[ok_login], dev[ok_login], ip[ok_login], "login")
    add(tt[~ok_login], cust[~ok_login], dev[~ok_login], ip[~ok_login], "login", success=False)
    # (c) device enrolment when a customer buys a new phone
    for k in (1, 2):
        m = np.nonzero(np.isfinite(world.dev_acq[:, k]) & (world.dev_acq[:, k] < T) & (world.dev_acq[:, k] > 0)
                       & (C["n_dev"][:nc] > k))[0]
        d_ = C["dev_start"][m] + k
        tm = world.dev_acq[m, k] + rng.uniform(0, 3600, len(m))
        add(tm, m, d_, home_ip(world, m, tm), "device_enrolment")
    # (d) password resets, email / phone changes of legit customers
    yr = cfg.n_days / 365
    for etype, p in (("password_reset", 0.06), ("phone_change", 0.04)):
        m = np.nonzero(rng.random(nc) < p * yr)[0]
        tm = open_t[m] + rng.uniform(0.02, 1, len(m)) * (T - open_t[m])
        add(tm, m, current_device(rng, world, m, tm), home_ip(world, m, tm), etype)
    m = np.nonzero(np.isfinite(C["email_change_t"][:nc]))[0]
    tm = C["email_change_t"][m]
    ok = tm > open_t[m]
    m, tm = m[ok], tm[ok]
    add(tm, m, current_device(rng, world, m, tm), home_ip(world, m, tm), "email_change", email=C["email2"][m])
    # (e) first payment to a new P2P payee
    p2p = np.nonzero((tx["channel"].to_numpy() == "P2P") & np.isin(scen, ["legit"]) & (pos["customer"] >= 0))[0]
    if len(p2p):
        pair = pd.DataFrame({"c": pos["customer"][p2p], "p": pos["counterparty"][p2p], "i": p2p})
        first = pair.groupby(["c", "p"])["i"].min().to_numpy()
        tm = pos["t"][first] - rng.uniform(30, 300, len(first))
        ok = tm >= 0
        first, tm = first[ok], tm[ok]
        add(tm, pos["customer"][first], pos["device"][first], pos["ip"][first], "new_payee",
            payee=pos["counterparty"][first])
    # (f) attackers (account takeover trail + credential stuffing noise)
    if ff.events:
        ev = pd.DataFrame(ff.events)
        ev = ev[(ev["t"] >= 0) & (ev["t"] < T)]
        for etype, g in ev.groupby("type"):
            em = g["email_domain"].to_numpy(dtype=object)
            add(g["t"].to_numpy(), g["customer"].to_numpy(), g["device"].to_numpy(), g["ip"].to_numpy(), etype,
                success=g["success"].to_numpy(), email=em, payee=g["counterparty"].to_numpy(), attacker=1,
                incident=g["incident"].to_numpy())
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("event_time", kind="stable").reset_index(drop=True)
    out.insert(0, "event_id", np.arange(1, len(out) + 1, dtype=np.int64))
    return out[EVENT_COLUMNS + EVENT_TRUTH_COLUMNS]

"""
Legitimate customer behaviour (fully vectorised).

* timing  : Poisson sessions per customer (gamma-distributed rates), weekly cycle,
            a personal peak hour mixed with a global diurnal curve (local time),
            sessions of 1+ purchases minutes apart (legit traffic is bursty too, so
            velocity alone must not separate fraud);
* where   : 55% habitual merchants, otherwise MCC by personal preference, local shops
            in the current city (home or trip destination) or online stores;
* amounts : lognormal per MCC, shifted by a personal spend level;
* identity: current device (new phones are acquired mid-period), home broadband /
            shared mobile CGNAT / VPN IPs, trips move IP and card-present location.
"""
from __future__ import annotations

import numpy as np

from . import geo
from .entities import World

CHUNK = 200_000


def _pick_mcc(rng, pref_cum: np.ndarray, cust: np.ndarray) -> np.ndarray:
    out = np.empty(len(cust), dtype=np.int64)
    n_mcc = pref_cum.shape[1]
    for s in range(0, len(cust), CHUNK):
        c = cust[s:s + CHUNK]
        out[s:s + CHUNK] = (pref_cum[c] < rng.random((len(c), 1))).sum(axis=1)
    return out.clip(0, n_mcc - 1)


def pick_terminal(rng, world: World, merchant: np.ndarray, card_present: np.ndarray) -> np.ndarray:
    M = world.merchants
    ts, nt, mode = M["term_start"][merchant], M["n_term"][merchant], M["mode"][merchant]
    u = rng.random(len(merchant))
    physical_first = np.where(mode == "omni", 1, 0)
    n_phys = np.maximum(nt - physical_first, 1)
    cp_term = ts + physical_first + np.floor(u * n_phys).astype(np.int64)
    return np.where(card_present, cp_term, ts)


def current_device(rng, world: World, cust: np.ndarray, t: np.ndarray, p_latest=0.75) -> np.ndarray:
    C = world.customers
    n_elig = (world.dev_acq[cust] <= t[:, None]).sum(axis=1).clip(1, None)
    n_elig = np.minimum(n_elig, C["n_dev"][cust])
    idx = np.where(rng.random(len(cust)) < p_latest, n_elig - 1,
                   np.floor(rng.random(len(cust)) * n_elig).astype(np.int64))
    return C["dev_start"][cust] + idx


def home_ip(world: World, cust: np.ndarray, t: np.ndarray) -> np.ndarray:
    C = world.customers
    k = np.floor((t + C["home_phase"][cust]) / C["home_period"][cust]).astype(np.int64)
    return C["home_start"][cust] + np.clip(k, 0, C["home_n"][cust] - 1)


def _pool_ip(rng, pools: dict, n_cities: int, city: np.ndarray) -> np.ndarray:
    mat = np.stack([pools[c] for c in range(n_cities)])
    return mat[city, rng.integers(0, mat.shape[1], len(city))]


def mobile_ip(rng, world: World, city: np.ndarray) -> np.ndarray:
    return _pool_ip(rng, world.mobile_pool, len(world.cities), city)


def generate_legit(world: World, rng: np.random.Generator) -> dict:
    cfg = world.cfg
    C, M, mccs = world.customers, world.merchants, world.mccs
    T = cfg.horizon_sec
    nc = cfg.n_customers
    open_t = C["open_t"][:nc]
    a0 = np.maximum(open_t, 0)
    active_days = (T - a0) / 86_400
    sess_mean = 4 / 3
    n_sess = rng.poisson(C["rate"][:nc] * active_days / sess_mean * 1.4)

    cust = np.repeat(np.arange(nc), n_sess)
    day = np.floor((a0[cust] + rng.random(len(cust)) * (T - a0[cust])) / 86_400).astype(np.int64)
    dow = (day + world.start_dow) % 7
    keep = rng.random(len(cust)) < np.where(dow >= 5, C["weekend_factor"][cust], 1.0) / 1.4
    cust, day = cust[keep], day[keep]
    n = len(cust)
    habit = rng.random(n) < 0.65
    hour = np.where(habit,
                    (C["peak_hour"][cust] + rng.normal(0, 1, n) * C["hour_spread"][cust]) % 24,
                    rng.choice(24, n, p=geo.GLOBAL_HOURLY) + rng.random(n))
    t = day * 86_400 + (hour * 3600).astype(np.int64) - (C["tz"][cust] * 3600).astype(np.int64)

    # sessions: 1 + Geometric extra purchases, minutes apart
    size = 1 + (rng.geometric(0.75, n) - 1)
    rep = np.repeat(np.arange(n), size)
    first = np.repeat(np.cumsum(size) - size, size)
    gap = rng.exponential(1200, len(rep))
    gap[np.arange(len(rep)) == first] = 0
    cs = np.cumsum(gap)
    t = t[rep] + (cs - cs[first]).astype(np.int64)
    cust = cust[rep]
    ok = (t >= np.maximum(open_t[cust], 0)) & (t >= 0) & (t < T)
    t, cust = t[ok], cust[ok]
    order = np.argsort(t, kind="stable")
    t, cust = t[order], cust[order]
    n = len(t)

    # trips
    tr = world.trips
    big = np.int64(1_000_000_000)
    k_tr = tr["cust"].to_numpy().astype(np.int64) * big + tr["start"].to_numpy().astype(np.int64)
    j = np.searchsorted(k_tr, cust * big + t, side="right") - 1
    jc = j.clip(0, max(len(tr) - 1, 0))
    if len(tr):
        in_trip = (j >= 0) & (tr["cust"].to_numpy()[jc] == cust) & (t < tr["end"].to_numpy()[jc])
        cur_city = np.where(in_trip, tr["dest_city"].to_numpy()[jc], C["city"][cust])
    else:
        in_trip = np.zeros(n, bool)
        cur_city = C["city"][cust]

    # channel, MCC, merchant
    online = C["online"][cust]
    has_contact = world.contacts[cust, 0] >= 0
    is_p2p = (rng.random(n) < C["p2p_share"][cust]) & has_contact & ~in_trip
    use_reg = ~is_p2p & ~in_trip & (rng.random(n) < 0.55)
    merchant = np.full(n, -1, dtype=np.int64)
    cnp = np.zeros(n, bool)
    cnp_share = mccs["cnp_share"].to_numpy()

    r = np.nonzero(use_reg)[0]
    merchant[r] = world.regulars[cust[r], rng.integers(0, 6, len(r))]
    mode_r = M["mode"][merchant[r]]
    p = np.clip(cnp_share[M["mcc_idx"][merchant[r]]] * online[r], 0, 1)
    far = M["city"][merchant[r]] != C["city"][cust[r]]
    cnp[r] = (mode_r == "cnp") | ((mode_r == "omni") & ((rng.random(len(r)) < p) | far))

    o = np.nonzero(~is_p2p & ~use_reg)[0]
    mcc_o = _pick_mcc(rng, world.pref_cum, cust[o])
    travel_mcc = np.nonzero(mccs["name"].isin(["hotels", "restaurants", "rideshare"]).to_numpy())[0]
    swap = in_trip[o] & (rng.random(len(o)) < 0.3)
    mcc_o[swap] = rng.choice(travel_mcc, swap.sum())
    cnp_o = rng.random(len(o)) < np.clip(cnp_share[mcc_o] * online[o], 0, 1)
    merchant[o], cnp[o] = world.picker.pick(rng, mcc_o, cur_city[o], cnp_o)

    pp = np.nonzero(is_p2p)[0]
    merchant[pp] = world.picker.p2p[cust[pp] % len(world.picker.p2p)]
    cnp |= M["mode"][merchant] == "cnp"
    cnp[pp] = False
    counterparty = np.full(n, -1, dtype=np.int64)
    n_valid = (world.contacts[cust[pp]] >= 0).sum(axis=1)
    counterparty[pp] = world.contacts[cust[pp], np.floor(rng.random(len(pp)) * n_valid).astype(np.int64)]

    channel = np.where(is_p2p, "P2P", np.where(cnp, "CNP", "CP")).astype(object)
    cp = channel == "CP"
    terminal = pick_terminal(rng, world, merchant, cp)

    pos = np.empty(n, dtype=object)
    pos[cp] = rng.choice(["chip", "contactless", "swipe", "keyed"], cp.sum(), p=[0.60, 0.33, 0.05, 0.02])
    is_cnp = channel == "CNP"
    pos[is_cnp] = rng.choice(["ecom", "card_on_file", "moto"], is_cnp.sum(), p=[0.75, 0.20, 0.05])
    pos[is_p2p] = "p2p"

    n_cards = C["n_cards"][cust]
    k = np.where((rng.random(n) < 0.8) | (n_cards == 1), 0,
                 1 + np.floor(rng.random(n) * (n_cards - 1)).astype(np.int64))
    card = C["card_start"][cust] + k

    mcc = M["mcc_idx"][merchant]
    mu = np.where(mcc >= 0, mccs["mu"].to_numpy()[mcc.clip(0)], 3.9) + C["spend_offset"][cust]
    sg = np.where(mcc >= 0, mccs["sigma"].to_numpy()[mcc.clip(0)], 0.9)
    value = np.exp(mu + sg * rng.normal(0, 1, n))

    # device & IP (online channels only)
    digital = ~cp
    device = np.full(n, -1, dtype=np.int64)
    d = np.nonzero(digital)[0]
    device[d] = current_device(rng, world, cust[d], t[d])
    p_null_dev = np.select([pos == "card_on_file", pos == "moto", pos == "ecom"], [0.8, 0.9, 0.12], 0.02)
    device[digital & (rng.random(n) < p_null_dev)] = -1

    ip = np.full(n, -1, dtype=np.int64)
    u = rng.random(len(d))
    ip_d = np.where(u < 0.66, home_ip(world, cust[d], t[d]), -1)
    mob = (u >= 0.66) & (u < 0.88)
    ip_d[mob] = mobile_ip(rng, world, C["city"][cust[d][mob]])
    wifi = (u >= 0.88) & (u < 0.94)
    ip_d[wifi] = _pool_ip(rng, world.wifi_pool, len(world.cities), C["city"][cust[d][wifi]])
    vpn = u >= 0.94
    ip_d[vpn] = rng.choice(world.vpn_pool, vpn.sum())
    trip_d = in_trip[d]
    ip_d[trip_d] = mobile_ip(rng, world, cur_city[d][trip_d])
    ip[d] = ip_d
    p_null_ip = np.select([pos == "card_on_file", pos == "moto"], [0.5, 1.0], 0.0)
    ip[digital & (rng.random(n) < p_null_ip)] = -1

    email = np.where(digital, C["email"][cust], None).astype(object)
    phys = np.where(mcc >= 0, mccs["physical"].to_numpy()[mcc.clip(0)], False)
    ship = np.full(n, -1, dtype=np.int64)
    s = is_cnp & phys
    ship[s] = np.where(rng.random(s.sum()) < 0.96, C["country"][cust[s]],
                       rng.integers(0, len(world.countries), s.sum()))
    cred = np.where(cp, "card_present", np.where(pos == "card_on_file", "card_on_file", "owner")).astype(object)

    return dict(t=t, customer=cust, card=card, merchant=merchant, terminal=terminal,
                counterparty=counterparty, channel=channel, pos_entry_mode=pos, value_usd=value,
                device=device, ip=ip, email_domain=email, shipping_country=ship, cred_type=cred)

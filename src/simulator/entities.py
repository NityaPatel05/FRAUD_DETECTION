"""
World entities: customers, cards, merchants, terminals, devices, IPs, households,
social contacts (for P2P) and trips.

Everything is held as column arrays (``Table``) indexed by position; public ids are
assigned only at the end (world.assemble) as random numbers so that an id never
encodes creation order (appended mule accounts / fraud devices would otherwise get
the highest ids — a leak).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from . import geo
from .config import SimConfig

NEG_INF = -1e15
POS_INF = 1e15


class Table:
    """Column store with cheap appends (chunks are concatenated lazily on read)."""

    def __init__(self, **cols):
        self._cols = {k: self._norm(np.asarray(v)) for k, v in cols.items()}
        lens = {len(v) for v in self._cols.values()}
        if len(lens) > 1:
            raise ValueError(f"columns have different lengths: {lens}")
        self._n = lens.pop() if lens else 0
        self._pending: list[dict] = []

    @staticmethod
    def _norm(a: np.ndarray) -> np.ndarray:
        return a.astype(object) if a.dtype.kind in "US" else a

    @staticmethod
    def _default(dtype):
        return {"f": np.nan, "i": -1, "b": False}.get(dtype.kind, None)

    def __len__(self) -> int:
        return self._n

    def _flush(self) -> None:
        if self._pending:
            for k in self._cols:
                self._cols[k] = np.concatenate([self._cols[k]] + [p[k] for p in self._pending])
            self._pending = []

    def __getitem__(self, key: str) -> np.ndarray:
        self._flush()
        return self._cols[key]

    def __setitem__(self, key: str, value) -> None:
        self._flush()
        value = self._norm(np.asarray(value))
        if len(value) != self._n:
            raise ValueError("length mismatch")
        self._cols[key] = value

    @property
    def columns(self):
        return list(self._cols)

    def append(self, **cols) -> np.ndarray:
        unknown = set(cols) - set(self._cols)
        if unknown:
            raise KeyError(f"unknown columns {unknown}")
        n = max((np.size(v) for v in cols.values() if np.ndim(v) > 0), default=1)
        chunk = {}
        for k, base in self._cols.items():
            if k in cols:
                v = np.asarray(cols[k])
                v = np.broadcast_to(v, (n,)) if v.ndim == 0 else v
                if len(v) != n:
                    raise ValueError(f"column {k}: length {len(v)} != {n}")
                chunk[k] = v.astype(base.dtype, copy=False) if base.dtype != object else v.astype(object)
            else:
                chunk[k] = np.full(n, self._default(base.dtype), dtype=base.dtype)
        self._pending.append(chunk)
        start = self._n
        self._n += n
        return np.arange(start, start + n)

    def to_frame(self) -> pd.DataFrame:
        self._flush()
        return pd.DataFrame(self._cols)


def _choice_by_group(rng, keys: np.ndarray, pools: Dict, n_default=None) -> np.ndarray:
    """For each row pick from pools[key] = (candidates, cumulative weights)."""
    out = np.full(len(keys), -1, dtype=np.int64)
    if len(keys) == 0:
        return out
    for k, pos in pd.Series(np.arange(len(keys))).groupby(keys).indices.items():
        pool = pools.get(k)
        if pool is None:
            continue
        cand, cw = pool
        out[pos] = cand[np.searchsorted(cw, rng.random(len(pos)) * cw[-1], side="right").clip(0, len(cand) - 1)]
    return out


def _pool(idx: np.ndarray, w: np.ndarray):
    return idx, np.cumsum(w)


class MerchantPicker:
    """Samples merchants by (MCC, card-present?, city) with popularity weights."""

    def __init__(self, merchants: Table, cities: pd.DataFrame):
        mode = merchants["mode"]
        mcc = merchants["mcc_idx"]
        city = merchants["city"]
        country = cities["country_idx"].to_numpy()[city]
        pop = merchants["popularity"]
        idx = np.arange(len(mode))
        shop = mcc >= 0
        online = shop & np.isin(mode, ["cnp", "omni"])
        local = shop & np.isin(mode, ["cp", "omni"])
        self.online = {m: _pool(idx[online & (mcc == m)], pop[online & (mcc == m)])
                       for m in np.unique(mcc[online])}
        self.local_city = {}
        self.local_country = {}
        self.local_any = {}
        df = pd.DataFrame({"i": idx[local], "mcc": mcc[local], "city": city[local],
                           "country": country[local], "pop": pop[local]})
        for (c, m), g in df.groupby(["city", "mcc"]):
            self.local_city[int(c) * 1000 + int(m)] = _pool(g["i"].to_numpy(), g["pop"].to_numpy())
        for (c, m), g in df.groupby(["country", "mcc"]):
            self.local_country[int(c) * 1000 + int(m)] = _pool(g["i"].to_numpy(), g["pop"].to_numpy())
        for m, g in df.groupby("mcc"):
            self.local_any[m] = _pool(g["i"].to_numpy(), g["pop"].to_numpy())
        self.city_country = cities["country_idx"].to_numpy()
        self.p2p = idx[mode == "p2p"]

    def pick(self, rng, mcc: np.ndarray, city: np.ndarray, cnp: np.ndarray):
        """Returns (merchant, card_not_present). A card-present request with no shop of
        that MCC in the city's country is served online instead (flag flips to True)."""
        mcc = np.asarray(mcc, dtype=np.int64)
        city = np.asarray(city, dtype=np.int64)
        cnp = np.asarray(cnp, dtype=bool)
        out = np.full(len(mcc), -1, dtype=np.int64)
        on = np.nonzero(cnp)[0]
        out[on] = _choice_by_group(rng, mcc[on], self.online)
        cp = np.nonzero(~cnp)[0]
        if len(cp):
            out[cp] = _choice_by_group(rng, city[cp] * 1000 + mcc[cp], self.local_city)
            # fallbacks: same country -> anywhere -> online
            miss = cp[out[cp] < 0]
            if len(miss):
                out[miss] = _choice_by_group(rng, self.city_country[city[miss]] * 1000 + mcc[miss],
                                             self.local_country)
            miss = cp[out[cp] < 0]
            if len(miss):
                out[miss] = _choice_by_group(rng, mcc[miss], self.online)
                cnp = cnp.copy()
                cnp[miss] = True
        miss = out < 0
        if miss.any():  # MCC with no merchant at all in this mode: any merchant of that MCC
            out[miss] = _choice_by_group(rng, mcc[miss], {**self.local_any, **self.online})
        return out, cnp


@dataclass
class World:
    cfg: SimConfig
    countries: pd.DataFrame
    cities: pd.DataFrame
    mccs: pd.DataFrame
    customers: Table
    cards: Table
    merchants: Table
    terminals: Table
    devices: Table
    ips: Table
    pref_cum: np.ndarray            # (n_customers, n_mcc) cumulative MCC preference
    regulars: np.ndarray            # (n_customers, 6) habitual merchants
    contacts: np.ndarray            # (n_customers, 5) P2P contacts, -1 = none
    dev_acq: np.ndarray             # (n_customers, 3) device acquisition times
    trips: pd.DataFrame
    mobile_pool: Dict[int, np.ndarray]
    wifi_pool: Dict[int, np.ndarray]
    vpn_pool: np.ndarray
    res_blocks: dict
    host_blocks: dict
    picker: MerchantPicker = None
    start_dow: int = 0
    bins_by_key: dict = field(default_factory=dict)

    # ── entity factories used by both legit and fraud generation ──
    def new_ips(self, rng, country: np.ndarray, city: np.ndarray, kind: str) -> np.ndarray:
        country = np.asarray(country, dtype=np.int64)
        city = np.asarray(city, dtype=np.int64)
        n = len(country)
        blocks = self.host_blocks if kind == "hosting" else self.res_blocks
        octet = np.empty(n, dtype=np.int64)
        for c in np.unique(country):
            pos = np.nonzero(country == c)[0]
            b = blocks.get(int(c))
            if b is None:  # no hosting block for this country -> use a US datacenter block
                b = blocks[0] if 0 in blocks else next(iter(blocks.values()))
            octet[pos] = rng.choice(b, len(pos))
        ip_int = (octet << 24) | rng.integers(0, 1 << 24, n)
        noise = {"residential": 0.06, "mobile": 0.25, "hosting": 0.01}[kind]
        cit = self.cities
        return self.ips.append(
            ip_int=ip_int, country=country, city=city,
            lat=cit["lat"].to_numpy()[city] + rng.normal(0, noise, n),
            lon=cit["lon"].to_numpy()[city] + rng.normal(0, noise, n),
            tz=cit["tz"].to_numpy()[city].astype(float),
            ip_type=np.full(n, kind, dtype=object))

    def new_devices(self, rng, n: int, owner: int | np.ndarray, kind: str, fraud: bool = False) -> np.ndarray:
        oses, p = geo.FRAUD_DEVICE_OS if fraud else geo.DEVICE_OS
        os_ = rng.choice(oses, n, p=p).astype(object)
        browser = np.empty(n, dtype=object)
        for o in np.unique(os_):
            pos = np.nonzero(os_ == o)[0]
            names, bp = geo.BROWSER_BY_OS[o]
            browser[pos] = rng.choice(names, len(pos), p=bp)
        return self.devices.append(owner=owner, owner_kind=kind, os=os_, browser=browser)

    def city_of_country(self, rng, country: np.ndarray) -> np.ndarray:
        cit = self.cities
        out = np.empty(len(country), dtype=np.int64)
        for c in np.unique(country):
            pos = np.nonzero(country == c)[0]
            cand = np.nonzero(cit["country_idx"].to_numpy() == c)[0]
            w = cit["w_in_country"].to_numpy()[cand]
            out[pos] = rng.choice(cand, len(pos), p=w / w.sum())
        return out

    def random_bins(self, rng, country: np.ndarray, network: np.ndarray, ctype: np.ndarray) -> np.ndarray:
        out = np.empty(len(country), dtype=np.int64)
        keys = pd.Series(list(zip(country, network, ctype)))
        codes, uni = pd.factorize(keys)
        for i, k in enumerate(uni):
            pos = np.nonzero(codes == i)[0]
            pool = self.bins_by_key.get(k)
            if pool is None:
                prefix = {"visa": 4, "mastercard": 5, "amex": 3, "discover": 6}[k[1]]
                pool = prefix * 100_000 + rng.choice(100_000, 6, replace=False)
                self.bins_by_key[k] = pool
            out[pos] = rng.choice(pool, len(pos))
        return out


def _weighted_city_global(countries, cities) -> np.ndarray:
    w = countries["customer_w"].to_numpy()[cities["country_idx"].to_numpy()] * cities["w_in_country"].to_numpy()
    return w / w.sum()


def build_world(cfg: SimConfig, rng: np.random.Generator) -> World:
    countries = geo.countries_df()
    cities = geo.cities_df(countries)
    mccs = geo.mccs_df()
    T = cfg.horizon_sec
    n_mcc = len(mccs)
    nc = cfg.n_customers
    hosting_countries = [int(i) for i in countries.index[countries["code"].isin(
        ["US", "DE", "GB", "FR", "RO", "RU", "CN", "JP", "BR", "IN"])]]
    res_blocks, host_blocks = geo.ip_blocks(len(countries), hosting_countries)

    # ── customers ──
    country = rng.choice(len(countries), nc, p=countries["customer_w"].to_numpy())
    world = World(cfg=cfg, countries=countries, cities=cities, mccs=mccs,
                  customers=None, cards=None, merchants=None, terminals=None,
                  devices=Table(owner=np.zeros(0, np.int64), owner_kind=np.zeros(0, object),
                                os=np.zeros(0, object), browser=np.zeros(0, object)),
                  ips=Table(ip_int=np.zeros(0, np.int64), country=np.zeros(0, np.int64),
                            city=np.zeros(0, np.int64), lat=np.zeros(0), lon=np.zeros(0),
                            tz=np.zeros(0), ip_type=np.zeros(0, object)),
                  pref_cum=None, regulars=None, contacts=None, dev_acq=None, trips=None,
                  mobile_pool={}, wifi_pool={}, vpn_pool=None, res_blocks=res_blocks, host_blocks=host_blocks,
                  start_dow=int(pd.Timestamp(cfg.start_date).dayofweek))
    city = world.city_of_country(rng, country)
    age0 = np.clip(rng.lognormal(np.log(900), 0.9, nc), 1, 9000)
    open_t = -age0 * 86_400
    new = rng.random(nc) < 0.08
    open_t[new] = rng.uniform(0, 0.9 * T, new.sum())
    email = rng.choice(geo.LEGIT_EMAIL[0], nc, p=geo.LEGIT_EMAIL[1]).astype(object)

    # households: ~25% of customers share their address with someone in the same city
    address = np.arange(nc, dtype=np.int64)
    share = rng.random(nc) < 0.25
    for c, pos in pd.Series(np.arange(nc)).groupby(city).indices.items():
        movers = pos[share[pos]]
        if len(movers) and len(pos) > 1:
            address[movers] = address[rng.choice(pos, len(movers))]

    # cards
    n_cards = rng.choice([1, 2, 3], nc, p=[0.65, 0.30, 0.05])
    card_owner = np.repeat(np.arange(nc), n_cards)
    card_start = np.concatenate([[0], np.cumsum(n_cards)[:-1]])
    ncard = len(card_owner)
    card_country = country[card_owner].copy()
    foreign = rng.random(ncard) < 0.03
    card_country[foreign] = rng.choice(len(countries), foreign.sum(), p=countries["customer_w"].to_numpy())
    network = rng.choice(["visa", "mastercard", "amex", "discover"], ncard, p=[0.55, 0.30, 0.10, 0.05]).astype(object)
    ctype = rng.choice(["debit", "credit", "prepaid"], ncard, p=[0.58, 0.37, 0.05]).astype(object)
    world.cards = Table(owner=card_owner, country=card_country, network=network, card_type=ctype,
                        bin=world.random_bins(rng, card_country, network, ctype))

    # devices: 1-3 per customer; extra devices are acquired during the simulation
    n_dev = 1 + np.minimum(rng.poisson(0.6, nc), 2)
    dev_owner = np.repeat(np.arange(nc), n_dev)
    world.new_devices(rng, len(dev_owner), dev_owner, "customer")
    dev_start = np.concatenate([[0], np.cumsum(n_dev)[:-1]])
    acq = np.full((nc, 3), POS_INF)
    acq[:, 0] = NEG_INF
    extra = np.sort(rng.uniform(0, T, (nc, 2)), axis=1)
    acq[:, 1] = np.where(n_dev >= 2, extra[:, 0], POS_INF)
    acq[:, 2] = np.where(n_dev >= 3, extra[:, 1], POS_INF)

    # IPs: home broadband with dynamic re-assignment every 20-150 days (15% static),
    # shared mobile CGNAT pools, public / work wifi pools per city, a shared VPN pool
    period = np.where(rng.random(nc) < 0.15, POS_INF, rng.uniform(20, 150, nc) * 86_400)
    home_n = np.where(np.isinf(period) | (period >= POS_INF), 1, np.ceil(T / np.minimum(period, T)).astype(np.int64) + 1)
    home_ips = world.new_ips(rng, np.repeat(country, home_n), np.repeat(city, home_n), "residential")
    home_start = home_ips[0] + np.concatenate([[0], np.cumsum(home_n)[:-1]])
    phase = rng.uniform(0, 1, nc) * np.minimum(period, T)
    for c in range(len(cities)):
        cc = int(cities["country_idx"].iat[c])
        world.mobile_pool[c] = world.new_ips(rng, np.full(40, cc), np.full(40, c), "mobile")
        world.wifi_pool[c] = world.new_ips(rng, np.full(300, cc), np.full(300, c), "residential")
    host_c = rng.choice(hosting_countries, 400)
    host_city = np.array([np.nonzero(cities["country_idx"].to_numpy() == c)[0][0] for c in host_c])
    world.vpn_pool = world.new_ips(rng, host_c, host_city, "hosting")

    world.customers = Table(
        country=country, city=city,
        home_lat=cities["lat"].to_numpy()[city] + rng.normal(0, 0.08, nc),
        home_lon=cities["lon"].to_numpy()[city] + rng.normal(0, 0.08, nc),
        tz=cities["tz"].to_numpy()[city].astype(float),
        open_t=open_t,
        rate=rng.gamma(1.2, cfg.tx_rate_mean / 1.2, nc),
        spend_offset=rng.normal(0.5, 0.35, nc),
        peak_hour=rng.normal(13.5, 3.0, nc),
        hour_spread=rng.uniform(2.0, 5.0, nc),
        weekend_factor=rng.uniform(0.6, 1.4, nc),
        online=rng.beta(2, 2, nc) * 2,
        p2p_share=np.where(rng.random(nc) < 0.25, rng.uniform(0.05, 0.2, nc), 0.0),
        email=email, address=address,
        card_start=card_start, n_cards=n_cards, dev_start=dev_start, n_dev=n_dev,
        home_start=home_start, home_n=home_n, home_period=period, home_phase=phase,
        kind=np.full(nc, "regular", dtype=object), is_mule=np.zeros(nc, bool),
    )
    world.dev_acq = acq

    # MCC preferences (Dirichlet around the global mix)
    pref = rng.dirichlet(25 * mccs["weight"].to_numpy() + 1e-3, size=nc)
    world.pref_cum = np.cumsum(pref, axis=1)

    # ── merchants & terminals ──
    nm = cfg.n_merchants
    w = mccs["weight"].to_numpy() ** 0.6
    mcc_idx = np.concatenate([np.repeat(np.arange(n_mcc), 3),
                              rng.choice(n_mcc, max(nm - 3 * n_mcc, 0), p=w / w.sum())])[:nm]
    cnp_share = mccs["cnp_share"].to_numpy()[mcc_idx]
    u = rng.random(len(mcc_idx))
    mode = np.where(cnp_share >= 0.95, "cnp", np.where(cnp_share <= 0.02, "cp",
                    np.where(u < cnp_share * 0.6, "cnp", np.where(u < cnp_share * 0.6 + 0.3, "omni", "cp")))).astype(object)
    m_city = rng.choice(len(cities), len(mcc_idx), p=_weighted_city_global(countries, cities))
    online_only = mode == "cnp"
    on_country = rng.choice(len(countries), online_only.sum(), p=countries["online_merchant_w"].to_numpy())
    m_city[online_only] = world.city_of_country(rng, on_country)
    # every city gets at least one local shop per everyday MCC, so card-present spend
    # never has to "teleport" to another country (that would fake impossible travel)
    everyday = np.nonzero(mccs["name"].isin(["grocery", "restaurants", "fuel", "pharmacy", "atm_cash",
                                             "department_store", "clothing", "electronics"]).to_numpy())[0]
    base_mcc = np.tile(everyday, len(cities))
    base_city = np.repeat(np.arange(len(cities)), len(everyday))
    mcc_idx = np.concatenate([base_mcc, mcc_idx])
    mode = np.concatenate([np.full(len(base_mcc), "cp", dtype=object), mode])
    m_city = np.concatenate([base_city, m_city])
    m_country = cities["country_idx"].to_numpy()[m_city]
    high_risk = mccs["high_risk"].to_numpy()[mcc_idx]
    uses_3ds = np.where(countries["sca"].to_numpy()[m_country], rng.random(len(mcc_idx)) < 0.97,
                        rng.random(len(mcc_idx)) < np.where(high_risk, 0.5, 0.25))
    # three P2P platforms (US)
    us_ny = int(np.nonzero(cities["city"].to_numpy() == "New York")[0][0])
    mcc_idx = np.concatenate([mcc_idx, np.full(3, -1)])
    mode = np.concatenate([mode, np.full(3, "p2p", dtype=object)])
    m_city = np.concatenate([m_city, np.full(3, us_ny)])
    uses_3ds = np.concatenate([uses_3ds, np.zeros(3, bool)])
    popularity = rng.lognormal(0, 1, len(mcc_idx))
    n_term = np.where(np.isin(mode, ["cp", "omni"]), 1 + np.minimum(rng.poisson(1.2, len(mode)), 7), 1)
    # omni merchants: terminal 0 is the virtual (e-commerce) terminal, the rest are physical
    n_term = np.where(mode == "omni", n_term + 1, n_term)
    term_start = np.concatenate([[0], np.cumsum(n_term)[:-1]])
    world.merchants = Table(mcc_idx=mcc_idx, mode=mode, city=m_city, uses_3ds=uses_3ds,
                            popularity=popularity, term_start=term_start, n_term=n_term)
    t_merchant = np.repeat(np.arange(len(mode)), n_term)
    t_rank = np.arange(len(t_merchant)) - term_start[t_merchant]
    virtual = (mode[t_merchant] == "cnp") | (mode[t_merchant] == "p2p") | ((mode[t_merchant] == "omni") & (t_rank == 0))
    world.terminals = Table(merchant=t_merchant, virtual=virtual)
    world.picker = MerchantPicker(world.merchants, cities)

    # habitual merchants: 6 per customer, local shops and online stores per MCC preference
    slot_mcc = (world.pref_cum[:, None, :] < rng.random((nc, 6, 1))).sum(axis=2).clip(0, n_mcc - 1)
    p_cnp = np.clip(mccs["cnp_share"].to_numpy()[slot_mcc] * world.customers["online"][:, None], 0, 1)
    slot_cnp = rng.random((nc, 6)) < p_cnp
    world.regulars = world.picker.pick(rng, slot_mcc.ravel(), np.repeat(city, 6), slot_cnp.ravel())[0].reshape(nc, 6)

    # P2P contacts: 0-5 people in the same city
    contacts = np.full((nc, 5), -1, dtype=np.int64)
    n_contacts = rng.integers(0, 6, nc)
    n_contacts[(world.customers["p2p_share"] > 0) & (n_contacts == 0)] = 2
    for c, pos in pd.Series(np.arange(nc)).groupby(city).indices.items():
        if len(pos) < 2:
            continue
        cand = rng.choice(pos, (len(pos), 5))
        cand[cand == pos[:, None]] = -1
        contacts[pos] = cand
    contacts[np.arange(5)[None, :] >= n_contacts[:, None]] = -1
    world.contacts = -np.sort(-contacts, axis=1)  # valid contacts first, -1 padding last

    # trips: 35% of customers travel ~1.5 times per half-year for 2-10 days
    traveller = rng.random(nc) < 0.35
    n_trips = np.where(traveller, rng.poisson(1.5 * cfg.n_days / 182, nc), 0)
    t_cust = np.repeat(np.arange(nc), n_trips)
    t_start = rng.uniform(0, T, len(t_cust))
    t_len = rng.uniform(2, 10, len(t_cust)) * 86_400
    foreign_trip = rng.random(len(t_cust)) < 0.35
    dest_country = np.where(foreign_trip,
                            rng.choice(len(countries), len(t_cust), p=countries["customer_w"].to_numpy()),
                            country[t_cust])
    dest = world.city_of_country(rng, dest_country)
    trips = pd.DataFrame({"cust": t_cust, "start": t_start, "end": t_start + t_len, "dest_city": dest})
    trips = trips[trips["dest_city"].to_numpy() != city[trips["cust"].to_numpy()]]
    trips = trips.sort_values(["cust", "start"]).reset_index(drop=True)
    # drop trips overlapping the customer's previous trip
    prev_end = trips.groupby("cust")["end"].shift(1)
    trips = trips[~(trips["start"] < prev_end)].reset_index(drop=True)
    world.trips = trips
    return world

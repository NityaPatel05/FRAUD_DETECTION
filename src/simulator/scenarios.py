"""
Fraud processes. Each scenario is an attacker agent that *generates* transactions;
``is_fraud`` / ``fraud_scenario`` record which process produced a row.

Scenarios (ground truth in ``fraud_scenario``):
  card_testing      bursts of tiny CNP charges on one merchant; stolen-list or
                    BIN-enumeration attacks (invalid PANs -> 14/54 declines); cards that
                    pass are added to the compromised pool for later cash-out.
  stolen_card_cnp   compromised card used online by a gang: gang devices / IPs / emails
                    reused across incidents, resale-friendly merchants, reshipping.
  account_takeover  attacker logs in from a new device + IP, changes email, buys with
                    the stored card and pushes P2P money to mule accounts.
  terminal_cashout  Handbook-style compromised terminal: a crew runs counterfeit /
                    stolen cards through one terminal for 1-4 weeks.
  skimming          cards used at a skimmed terminal are cloned and used card-present
                    in another city weeks later (common point of purchase).
  mule_ring         AMLworld-style laundering: scam victims push money to entry mules,
                    funds hop through the ring (chain / scatter-gather / gather-scatter /
                    cycle / fan-out), exits cash out. ``ring_id`` is ground truth.
  bust_out          NOVEL scenario, test period only: long-standing accounts suddenly
                    max out on high-value goods from their usual device and IP.
  friendly_fraud    (world.py) legit purchases later disputed: deliberate label noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import geo
from .entities import World, _weighted_city_global
from .legit import current_device, home_ip, pick_terminal
from .rows import Rows

HOUR, DAY = 3600, 86_400


class FraudFactory:
    def __init__(self, world: World, legit: dict, rng: np.random.Generator):
        self.w = world
        self.rng = rng
        self.cfg = world.cfg
        self.T = self.cfg.horizon_sec
        self.rows = Rows(horizon=self.T)
        self.incidents: list[dict] = []
        self.rings: list[dict] = []
        self.ring_members: list[dict] = []
        self.pool: list[tuple[int, int]] = []        # (card, time compromised)
        self.bust_out_end: dict[int, int] = {}       # customer -> time after which legit stops
        self.n_cust0 = self.cfg.n_customers
        self.n_card0 = len(world.cards)
        self.next_address = int(world.customers["address"].max()) + 1
        self.legit = legit

        M, mccs = world.merchants, world.mccs
        name_of = dict(enumerate(mccs["name"]))
        self.mcc_by_name = {v: k for k, v in name_of.items()}
        mcc = M["mcc_idx"]
        mode = M["mode"]
        high = np.zeros(len(mcc), bool)
        high[mcc >= 0] = mccs["high_risk"].to_numpy()[mcc[mcc >= 0]]
        self.online = np.nonzero(np.isin(mode, ["cnp", "omni"]))[0]
        self.online_high = np.nonzero(np.isin(mode, ["cnp", "omni"]) & high)[0]
        m_country = world.cities["country_idx"].to_numpy()[M["city"]]
        # gangs favour cross-border stores (weaker issuer checks, easier resale)
        w_high = np.where(m_country[self.online_high] != 0, 6.0, 1.0)
        self.online_high_p = w_high / w_high.sum()
        test_mccs = [self.mcc_by_name[n] for n in ("donations", "digital_goods", "misc_retail", "streaming")]
        self.test_merchants = np.nonzero(np.isin(mode, ["cnp", "omni"]) & np.isin(mcc, test_mccs))[0]

        T_ = world.terminals
        phys = np.nonzero(~T_["virtual"])[0]
        t_mcc = mcc[T_["merchant"][phys]]
        cash_m = [self.mcc_by_name[n] for n in ("electronics", "jewelry", "department_store", "grocery", "fuel", "misc_retail")]
        skim_m = [self.mcc_by_name[n] for n in ("fuel", "atm_cash", "grocery")]
        self.cashout_terminals = phys[np.isin(t_mcc, cash_m)]
        self.skim_terminals = phys[np.isin(t_mcc, skim_m)]

        cards = world.cards
        self.cards_by_bin = pd.Series(np.arange(self.n_card0)).groupby(cards["bin"][:self.n_card0]).indices

        # legit rows sorted by terminal for skimming
        key = legit["terminal"].astype(np.int64) * np.int64(1_000_000_000) + legit["t"]
        self.leg_order = np.argsort(key, kind="stable")
        self.leg_key = key[self.leg_order]

        self._make_gangs()

    # ── helpers ──
    def _new_incident(self, scenario: str, **info) -> int:
        iid = len(self.incidents)
        self.incidents.append({"incident": iid, "scenario": scenario, **info})
        return iid

    def _delay(self, n, median, lo, hi):
        return np.clip(self.rng.lognormal(np.log(median), 0.6, n), lo, hi)

    def _mu_sigma(self, merchant):
        mcc = self.w.merchants["mcc_idx"][merchant]
        m = self.w.mccs
        return (np.where(mcc >= 0, m["mu"].to_numpy()[mcc.clip(0)], 4.5),
                np.where(mcc >= 0, m["sigma"].to_numpy()[mcc.clip(0)], 0.8))

    def _make_gangs(self):
        w, rng = self.w, self.rng
        cty = w.countries
        us_cities = np.nonzero(w.cities["country_idx"].to_numpy() == 0)[0]
        self.gangs = []
        for g in range(self.cfg.n_gangs):
            gc = int(rng.choice(len(cty), p=cty["fraud_home_w"].to_numpy()))
            devices = w.new_devices(rng, int(rng.integers(3, 16)), -1, "gang", fraud=True)
            n_h = int(rng.integers(3, 16))
            shared = rng.choice(w.vpn_pool, n_h // 2)
            own_h = w.new_ips(rng, np.full(n_h - n_h // 2, gc), w.city_of_country(rng, np.full(n_h - n_h // 2, gc)), "hosting")
            n_p = int(rng.integers(3, 16))
            proxy = w.new_ips(rng, np.zeros(n_p, np.int64), rng.choice(us_cities, n_p), "residential")
            n_r = int(rng.integers(2, 9))
            home = w.new_ips(rng, np.full(n_r, gc), w.city_of_country(rng, np.full(n_r, gc)), "residential")
            self.gangs.append({
                "country": gc, "devices": devices,
                "ips": {"hosting": np.concatenate([shared, own_h]), "proxy": proxy, "home": home},
                "emails": rng.choice(geo.FRAUD_EMAIL[0], 3, p=geo.FRAUD_EMAIL[1]),
                "merchants": rng.choice(self.online_high, int(rng.integers(10, 41)), p=self.online_high_p),
            })
        wts = rng.pareto(1.2, self.cfg.n_gangs) + 1
        self.gang_p = wts / wts.sum()

    def _gang(self):
        g = int(self.rng.choice(len(self.gangs), p=self.gang_p))
        return g, self.gangs[g]

    def _gang_ips(self, gang, n, p_host, p_proxy):
        u = self.rng.random(n)
        ips = gang["ips"]
        return np.where(u < p_host, self.rng.choice(ips["hosting"], n),
                        np.where(u < p_host + p_proxy, self.rng.choice(ips["proxy"], n),
                                 self.rng.choice(ips["home"], n)))

    def _attack_ip(self, gang, victim, p_host, p_proxy) -> int:
        u = self.rng.random()
        if p_host <= u < p_host + p_proxy and victim >= 0 and self.rng.random() < 0.5:
            C = self.w.customers
            return int(self.w.new_ips(self.rng, np.array([C["country"][victim]]),
                                      np.array([C["city"][victim]]), "residential")[0])
        return int(self._gang_ips(gang, 1, p_host, p_proxy)[0])

    def _cust_ok(self, cust, t):
        C = self.w.customers
        return (cust >= 0) and (C["open_t"][cust] < t) and (self.bust_out_end.get(int(cust), 1e18) > t)

    def _random_customer(self, t, exclude_mules=True):
        C = self.w.customers
        for _ in range(50):
            c = int(self.rng.integers(0, self.n_cust0))
            if self._cust_ok(c, t) and not (exclude_mules and C["is_mule"][c]):
                return c
        return int(self.rng.integers(0, self.n_cust0))

    # ── scenarios: each makes ONE incident and returns the number of rows ──
    def card_testing(self) -> int:
        w, rng, T = self.w, self.rng, self.T
        cards = w.cards
        g, gang = self._gang()
        m = int(rng.choice(self.test_merchants))
        dur = float(np.exp(rng.uniform(np.log(600), np.log(6 * HOUR))))
        t0 = rng.uniform(-dur, T)
        n_att = int(np.exp(rng.uniform(np.log(20), np.log(300))))
        n_cards = max(1, int(n_att / rng.uniform(1, 2.5)))
        enumerated = rng.random() < 0.4
        if enumerated:
            ref = int(rng.integers(0, self.n_card0))
            b = int(cards["bin"][ref])
            cand = self.cards_by_bin.get(b, np.array([ref]))
            k_valid = min(rng.binomial(n_cards, rng.uniform(0.01, 0.05)), len(cand))
            valid = rng.choice(cand, k_valid, replace=False)
            n_fake = n_cards - k_valid
            fake = cards.append(owner=-1, country=np.full(n_fake, cards["country"][ref]),
                                network=np.full(n_fake, cards["network"][ref], dtype=object),
                                card_type=np.full(n_fake, cards["card_type"][ref], dtype=object),
                                bin=np.full(n_fake, b))
            valid_cred = np.full(k_valid, "enumerated_valid", dtype=object)
        else:
            k_valid = rng.binomial(n_cards, rng.uniform(0.15, 0.5))
            valid = rng.choice(self.n_card0, k_valid, replace=False)
            n_fake = n_cards - k_valid
            ctry = rng.choice(len(w.countries), n_fake, p=w.countries["customer_w"].to_numpy())
            net = rng.choice(["visa", "mastercard", "amex", "discover"], n_fake, p=[0.55, 0.3, 0.1, 0.05]).astype(object)
            typ = rng.choice(["debit", "credit", "prepaid"], n_fake, p=[0.55, 0.35, 0.10]).astype(object)
            fake = cards.append(owner=-1, country=ctry, network=net, card_type=typ,
                                bin=w.random_bins(rng, ctry, net, typ))
            valid_cred = rng.choice(["stolen_full", "stolen_partial"], k_valid, p=[0.5, 0.5]).astype(object)
        card_list = np.concatenate([valid, fake]).astype(np.int64)
        cred_list = np.concatenate([valid_cred, np.full(n_fake, "invalid_pan", dtype=object)])
        att = np.concatenate([np.arange(n_cards), rng.integers(0, n_cards, max(n_att - n_cards, 0))])
        rng.shuffle(att)
        n = len(att)
        t = (t0 + np.sort(rng.random(n)) * dur).astype(np.int64)
        card = card_list[att]
        owner = cards["owner"][card]
        devs = rng.choice(gang["devices"], int(rng.integers(1, 3)))
        device = rng.choice(devs, n) if rng.random() > 0.6 else np.full(n, -1)
        value = np.where(rng.random(n) < 0.3, 1.0, rng.uniform(0.5, 5.0, n))
        iid = self._new_incident("card_testing", gang=g, start=int(t[0]), end=int(t[-1]), merchant=m,
                                 enumerated=bool(enumerated), n_cards=int(n_cards), n_valid=int(len(valid)))
        kept = self.rows.add(n, t=t, customer=owner, card=card, merchant=m, terminal=w.merchants["term_start"][m],
                      channel="CNP", pos_entry_mode="ecom", value_usd=value, device=device,
                      ip=self._gang_ips(gang, n, 0.6, 0.4),
                      email_domain=rng.choice(geo.FRAUD_EMAIL[0], n, p=geo.FRAUD_EMAIL[1]).astype(object),
                      cred_type=cred_list[att], is_fraud=1, scenario="card_testing", incident=iid,
                      delay_days=rng.uniform(1, 10, n))
        last_t = pd.Series(t).groupby(card).max()
        for c in valid:
            self.pool.append((int(c), int(last_t.get(c, t0))))
        return kept

    def stolen_card_cnp(self) -> int:
        w, rng, T = self.w, self.rng, self.T
        cards, M = w.cards, w.merchants
        g, gang = self._gang()
        if self.pool and rng.random() < 0.5:
            card, tc = self.pool.pop(int(rng.integers(0, len(self.pool))))
            t0 = tc + rng.uniform(0.5, 20) * DAY
        else:
            card = int(rng.integers(0, self.n_card0))
            t0 = rng.uniform(-2 * DAY, T)
        owner = int(cards["owner"][card])
        if t0 >= T or not self._cust_ok(owner, max(t0, 0)):
            return 0
        n = int(min(1 + rng.geometric(0.35), 10))
        t = (t0 + np.sort(rng.random(n)) * rng.uniform(2 * HOUR, 4 * DAY)).astype(np.int64)
        merchant = np.where(rng.random(n) < 0.5, rng.choice(gang["merchants"], n), rng.choice(self.online, n))
        mu, sg = self._mu_sigma(merchant)
        value = np.exp(mu + 0.4 + sg * rng.normal(0, 1, n))
        if rng.random() < 0.3:
            value[0] = rng.uniform(1, 10)
        dev = int(rng.choice(gang["devices"])) if rng.random() < 0.75 else int(w.new_devices(rng, 1, -1, "gang", True)[0])
        ip = rng.choice([self._attack_ip(gang, owner, 0.35, 0.25) for _ in range(2)], n)
        mcc = M["mcc_idx"][merchant]
        phys = w.mccs["physical"].to_numpy()[mcc.clip(0)] & (mcc >= 0)
        ship = np.where(phys, np.where(rng.random(n) < 0.55, cards["country"][card], gang["country"]), -1)
        cred = "stolen_full" if rng.random() < 0.6 else "stolen_partial"
        iid = self._new_incident("stolen_card_cnp", gang=g, start=int(t[0]), end=int(t[-1]), card=card)
        kept = self.rows.add(n, t=t, customer=owner, card=card, merchant=merchant,
                      terminal=M["term_start"][merchant], channel="CNP", pos_entry_mode="ecom",
                      value_usd=value, device=dev, ip=ip, email_domain=rng.choice(gang["emails"]),
                      shipping_country=ship, cred_type=cred, is_fraud=1, scenario="stolen_card_cnp",
                      incident=iid, delay_days=self._delay(n, 35, 7, 120))
        return kept

    def account_takeover(self) -> int:
        w, rng, T = self.w, self.rng, self.T
        C, M = w.customers, w.merchants
        g, gang = self._gang()
        t0 = rng.uniform(-DAY, T)
        victim = -1
        for _ in range(50):
            c = int(rng.integers(0, self.n_cust0))
            if C["online"][c] > 1.0 and self._cust_ok(c, max(t0, 0)) and not C["is_mule"][c]:
                victim = c
                break
        if victim < 0:
            return 0
        card = int(C["card_start"][victim])
        dev = int(rng.choice(gang["devices"])) if rng.random() < 0.6 else int(w.new_devices(rng, 1, -1, "gang", True)[0])
        ip = self._attack_ip(gang, victim, 0.3, 0.4)
        email = rng.choice(gang["emails"])
        n_p = int(rng.integers(1, 6))
        n_x = int(rng.integers(0, 4)) if self.rings else 0
        n = n_p + n_x
        t = (t0 + np.sort(rng.random(n)) * rng.uniform(2 * HOUR, 2 * DAY)).astype(np.int64)
        is_x = np.zeros(n, bool)
        is_x[rng.choice(n, n_x, replace=False)] = True
        merchant = np.where(is_x, w.picker.p2p[victim % len(w.picker.p2p)], rng.choice(self.online_high, n))
        mu, sg = self._mu_sigma(merchant)
        value = np.where(is_x, rng.lognormal(5.9, 0.6, n), np.exp(mu + 0.5 + sg * rng.normal(0, 1, n)))
        counterparty = np.full(n, -1, dtype=np.int64)
        ring = np.full(n, -1, dtype=np.int64)
        if n_x:
            active = [r for r in self.rings if r["start"] <= t0 <= r["end"]] or self.rings
            r = active[int(rng.integers(0, len(active)))]
            counterparty[is_x] = rng.choice(r["entries"], n_x)
            ring[is_x] = r["ring"]
        mcc = M["mcc_idx"][merchant]
        phys = (mcc >= 0) & w.mccs["physical"].to_numpy()[mcc.clip(0)]
        ship = np.where(phys, np.where(rng.random(n) < 0.5, C["country"][victim], gang["country"]), -1)
        iid = self._new_incident("account_takeover", gang=g, start=int(t[0]), end=int(t[-1]), customer=victim)
        kept = self.rows.add(n, t=t, customer=victim, card=card, merchant=merchant, terminal=M["term_start"][merchant],
                      counterparty=counterparty, channel=np.where(is_x, "P2P", "CNP").astype(object),
                      pos_entry_mode=np.where(is_x, "p2p", "card_on_file").astype(object), value_usd=value,
                      device=dev, ip=ip, email_domain=email, shipping_country=ship,
                      cred_type=np.where(is_x, "account_login", "card_on_file").astype(object),
                      is_fraud=1, scenario="account_takeover", incident=iid, ring=ring,
                      delay_days=self._delay(n, 20, 2, 90))
        return kept

    def terminal_cashout(self) -> int:
        w, rng, T = self.w, self.rng, self.T
        cards, M = w.cards, w.merchants
        if not len(self.cashout_terminals):
            return 0
        term = int(rng.choice(self.cashout_terminals))
        m = int(w.terminals["merchant"][term])
        W = rng.uniform(7, 28) * DAY
        t0 = rng.uniform(-W, T)
        n = int(rng.poisson(rng.uniform(2, 6) * W / DAY))
        if n == 0:
            return 0
        n_cards = max(1, int(n / 1.5))
        from_pool = [i for i, (c, tc) in enumerate(self.pool) if tc < t0][: n_cards // 2]
        pool_cards = [self.pool[i][0] for i in from_pool]
        for i in sorted(from_pool, reverse=True):
            self.pool.pop(i)
        uniq = np.concatenate([np.array(pool_cards, dtype=np.int64),
                               rng.integers(0, self.n_card0, n_cards - len(pool_cards))])
        card = np.concatenate([uniq, rng.choice(uniq, max(n - len(uniq), 0))])[:n]
        t = (t0 + np.sort(rng.random(n)) * W).astype(np.int64)
        mu, sg = self._mu_sigma(np.full(n, m))
        iid = self._new_incident("terminal_cashout", start=int(t[0]), end=int(t[-1]), terminal=term, merchant=m)
        kept = self.rows.add(n, t=t, customer=cards["owner"][card], card=card, merchant=m, terminal=term,
                      channel="CP",
                      pos_entry_mode=rng.choice(["swipe", "keyed", "chip"], n, p=[0.55, 0.25, 0.20]).astype(object),
                      value_usd=np.exp(mu + 0.6 + sg * rng.normal(0, 1, n)), cred_type="counterfeit",
                      is_fraud=1, scenario="terminal_cashout", incident=iid,
                      delay_days=self._delay(n, 40, 10, 120))
        return kept

    def skimming(self) -> int:
        w, rng, T = self.w, self.rng, self.T
        cards, cities = w.cards, w.cities
        if not len(self.skim_terminals):
            return 0
        term = int(rng.choice(self.skim_terminals))
        W = rng.uniform(5, 30) * DAY
        t0 = rng.uniform(-W - 15 * DAY, T)
        big = np.int64(1_000_000_000)
        lo = np.searchsorted(self.leg_key, term * big + int(t0))
        hi = np.searchsorted(self.leg_key, term * big + int(t0 + W))
        rows = self.leg_order[lo:hi]
        if not len(rows):
            return 0
        skim_card = self.legit["card"][rows]
        skim_t = self.legit["t"][rows]
        first = pd.Series(skim_t).groupby(skim_card).min()
        k = rng.binomial(len(first), 0.6)
        if k == 0:
            return 0
        chosen = rng.choice(len(first), k, replace=False)
        c_cards = first.index.to_numpy()[chosen].astype(np.int64)
        c_t = first.to_numpy()[chosen]
        term_city = int(w.merchants["city"][w.terminals["merchant"][term]])
        if rng.random() < 0.6:
            same = np.nonzero((cities["country_idx"].to_numpy() == cities["country_idx"].iat[term_city])
                              & (np.arange(len(cities)) != term_city))[0]
            crew_city = int(rng.choice(same)) if len(same) else term_city
        else:
            fc = int(rng.choice(len(w.countries), p=w.countries["fraud_home_w"].to_numpy()))
            crew_city = int(w.city_of_country(rng, np.array([fc]))[0])
        n_use = rng.integers(1, 5, k)
        card = np.repeat(c_cards, n_use)
        t = (np.repeat(c_t + rng.uniform(2, 30, k) * DAY, n_use) + rng.uniform(0, 2 * DAY, len(card))).astype(np.int64)
        n = len(card)
        names = ["electronics", "jewelry", "department_store", "atm_cash", "gift_cards"]
        mcc = rng.choice([self.mcc_by_name[x] for x in names], n, p=[0.3, 0.15, 0.2, 0.25, 0.1])
        merchant, online = w.picker.pick(rng, mcc, np.full(n, crew_city), np.zeros(n, bool))
        keep = ~online  # counterfeit stripes only work card-present
        card, t, merchant = card[keep], t[keep], merchant[keep]
        n = len(card)
        if n == 0:
            return 0
        terminal = pick_terminal(rng, w, merchant, np.ones(n, bool))
        mu, sg = self._mu_sigma(merchant)
        iid = self._new_incident("skimming", start=int(t.min()), end=int(t.max()), terminal=term,
                                 skim_start=int(t0), skim_end=int(t0 + W), n_cards=int(k))
        kept = self.rows.add(n, t=t, customer=cards["owner"][card], card=card, merchant=merchant, terminal=terminal,
                      channel="CP", pos_entry_mode=rng.choice(["swipe", "keyed"], n, p=[0.8, 0.2]).astype(object),
                      value_usd=np.exp(mu + 0.5 + sg * rng.normal(0, 1, n)), cred_type="counterfeit",
                      is_fraud=1, scenario="skimming", incident=iid, delay_days=self._delay(n, 30, 10, 120))
        return kept

    def mule_ring(self) -> int:
        w, rng, T = self.w, self.rng, self.T
        C, cities = w.customers, w.cities
        m = int(rng.integers(3, 13))
        dur = rng.uniform(3, 21) * DAY
        t0 = rng.uniform(-dur, T)
        ring_id = len(self.rings)
        ring_city = int(rng.choice(len(cities), p=_weighted_city_global(w.countries, cities)))
        ring_country = int(cities["country_idx"].iat[ring_city])
        n_new = rng.binomial(m, 0.5)
        recruited = [self._random_customer(t0) for _ in range(m - n_new)]
        r_dev = w.new_devices(rng, int(rng.integers(1, 4)), -1, "ring")
        n_ip = int(rng.integers(1, 4))
        r_ip = w.new_ips(rng, np.full(n_ip, ring_country), np.full(n_ip, ring_city), "residential")
        new_ids = np.zeros(0, dtype=np.int64)
        if n_new:
            open_t = t0 - rng.uniform(1, 60, n_new) * DAY
            new_ids = C.append(country=ring_country, city=ring_city,
                               home_lat=cities["lat"].iat[ring_city] + rng.normal(0, 0.08, n_new),
                               home_lon=cities["lon"].iat[ring_city] + rng.normal(0, 0.08, n_new),
                               tz=float(cities["tz"].iat[ring_city]), open_t=open_t, rate=0.0,
                               spend_offset=0.0, online=1.0, p2p_share=0.0,
                               email=rng.choice(geo.LEGIT_EMAIL[0], n_new, p=geo.LEGIT_EMAIL[1]).astype(object),
                               address=self.next_address + np.arange(n_new), n_cards=1, n_dev=0,
                               home_start=rng.choice(r_ip, n_new), home_n=1, home_period=1e15,
                               home_phase=0.0, kind="mule_new", is_mule=True)
            self.next_address += n_new
            new_cards = w.cards.append(owner=new_ids, country=ring_country, network="visa", card_type="debit",
                                       bin=w.random_bins(rng, np.full(n_new, ring_country),
                                                         np.full(n_new, "visa", dtype=object),
                                                         np.full(n_new, "debit", dtype=object)))
            cs = C["card_start"]
            cs[new_ids] = new_cards
        members = np.concatenate([np.array(recruited, dtype=np.int64), new_ids])
        rng.shuffle(members)
        mule = C["is_mule"]
        mule[members] = True
        addr = C["address"]
        share = members[rng.random(len(members)) < 0.4]
        if len(share):
            addr[share] = self.next_address
            self.next_address += 1

        pattern = str(rng.choice(["chain", "scatter_gather", "gather_scatter", "cycle", "fan_out"]))
        k = len(members)
        if pattern == "chain":
            entries, L = members[:1], min(k, 5)
            stages = [[(members[i], members[i + 1], 1.0)] for i in range(L - 1)]
            exits = members[L - 1:L]
        elif pattern == "scatter_gather":
            mid = members[1:max(2, k - 1)][:5]
            entries, exits = members[:1], members[-1:]
            stages = [[(members[0], x, 1 / len(mid)) for x in mid], [(x, members[-1], 1.0) for x in mid]]
        elif pattern == "gather_scatter":
            a = max(1, min(3, k - 2))
            entries, hub, exits = members[:a], members[a], members[a + 1:] if k > a + 1 else members[a:a + 1]
            stages = [[(e, hub, 1.0) for e in entries]] + ([[(hub, x, 1 / len(exits)) for x in exits]]
                                                          if exits[0] != hub else [])
        elif pattern == "cycle":
            L = min(k, 5)
            entries, exits = members[:1], members[:1]
            stages = [[(members[i], members[(i + 1) % L], 1.0)] for i in range(L)]
        else:  # fan_out
            entries, exits = members[:1], members[1:]
            stages = [[(members[0], x, 1 / len(exits)) for x in exits]]

        p2p = w.picker.p2p
        rows = {k_: [] for k_ in ("t", "customer", "counterparty", "value", "device", "ip", "kind", "merchant", "channel")}

        def add(t_, cust, cp, val, dev, ip_, kind, merchant, channel="P2P"):
            rows["t"].append(int(t_)); rows["customer"].append(int(cust)); rows["counterparty"].append(int(cp))
            rows["value"].append(float(val)); rows["device"].append(int(dev)); rows["ip"].append(int(ip_))
            rows["kind"].append(kind); rows["merchant"].append(int(merchant)); rows["channel"].append(channel)

        R = int(rng.integers(1, 5))
        for r in range(R):
            tr = t0 + r * dur / R + rng.uniform(0, 0.3 * dur / R)
            bal = {int(x): 0.0 for x in members}
            ready = {int(x): tr for x in members}
            for _ in range(int(rng.integers(2, 9))):
                vt = tr + rng.uniform(0, DAY)
                victim = self._random_customer(vt)
                e = int(rng.choice(entries))
                amt = float(rng.lognormal(5.6, 0.7))
                dev = int(current_device(rng, w, np.array([victim]), np.array([int(vt)]))[0]) if victim < self.n_cust0 else -1
                ip_ = int(home_ip(w, np.array([victim]), np.array([int(vt)]))[0])
                add(vt, victim, e, amt, dev, ip_, "victim", p2p[victim % len(p2p)])
                bal[e] += amt
                ready[e] = max(ready[e], vt)
            for stage in stages:
                snap = dict(bal)
                for src, dst, frac in stage:
                    src, dst = int(src), int(dst)
                    amt = snap[src] * frac * (1 - rng.uniform(0.02, 0.10))
                    if amt < 1:
                        continue
                    ts = ready[src] + rng.exponential(4 * HOUR)
                    add(ts, src, dst, amt, rng.choice(r_dev), rng.choice(r_ip), "layer", p2p[src % len(p2p)])
                    bal[src] -= snap[src] * frac
                    bal[dst] += amt
                    ready[dst] = max(ready[dst], ts)
            for x in exits:
                x = int(x)
                remaining = bal[x]
                if remaining < 20:
                    continue
                n_co = int(rng.integers(1, 5))
                parts = rng.dirichlet(np.ones(n_co)) * remaining * 0.95
                ts = ready[x]
                for amt in parts:
                    ts += rng.exponential(6 * HOUR)
                    u = rng.random()
                    if u < 0.4:
                        mcc, cnp = self.mcc_by_name["crypto_quasi_cash"], True
                    elif u < 0.7:
                        mcc, cnp = self.mcc_by_name["gift_cards"], True
                    else:
                        mcc, cnp = self.mcc_by_name["atm_cash"], False
                    m_arr, c_arr = w.picker.pick(rng, np.array([mcc]), np.array([ring_city]), np.array([cnp]))
                    merch, cnp = int(m_arr[0]), bool(c_arr[0])
                    add(ts, x, -1, amt, rng.choice(r_dev) if cnp else -1, rng.choice(r_ip) if cnp else -1,
                        "cashout", merch, "CNP" if cnp else "CP")
                bal[x] = 0.0

        n = len(rows["t"])
        if n == 0:
            return 0
        t = np.array(rows["t"], dtype=np.int64)
        cust = np.array(rows["customer"], dtype=np.int64)
        merchant = np.array(rows["merchant"], dtype=np.int64)
        channel = np.array(rows["channel"], dtype=object)
        kind = np.array(rows["kind"], dtype=object)
        cp_mask = channel == "CP"
        terminal = pick_terminal(rng, w, merchant, cp_mask)
        pos = np.where(channel == "P2P", "p2p", np.where(cp_mask, "chip", "ecom")).astype(object)
        cred = np.where(cp_mask, "card_present", "owner").astype(object)
        iid = self._new_incident("mule_ring", start=int(t.min()), end=int(t.max()), ring=ring_id, pattern=pattern)
        kept = self.rows.add(n, t=t, customer=cust, card=C["card_start"][cust], merchant=merchant, terminal=terminal,
                      counterparty=np.array(rows["counterparty"], dtype=np.int64), channel=channel,
                      pos_entry_mode=pos, value_usd=np.array(rows["value"]),
                      device=np.array(rows["device"], dtype=np.int64), ip=np.array(rows["ip"], dtype=np.int64),
                      email_domain=np.where(cp_mask, None, C["email"][cust]).astype(object),
                      cred_type=cred, is_fraud=1, scenario="mule_ring", incident=iid, ring=ring_id,
                      delay_days=np.where(kind == "victim", rng.uniform(1, 30, n), rng.uniform(5, 60, n)))
        self.rings.append({"ring": ring_id, "pattern": pattern, "start": int(t.min()), "end": int(t.max()),
                           "city": ring_city, "n_members": int(len(members)), "entries": np.asarray(entries),
                           "incident": iid})
        new_set = set(new_ids.tolist())
        roles = {int(x): "middle" for x in members}
        for x in exits:
            roles[int(x)] = "exit"
        for x in entries:
            roles[int(x)] = "entry" if roles[int(x)] != "exit" else "entry_exit"
        for x in members:
            self.ring_members.append({"ring": ring_id, "customer": int(x), "role": roles[int(x)],
                                      "new_account": int(x) in new_set})
        return kept

    def bust_out(self) -> int:
        w, rng, cfg = self.w, self.rng, self.cfg
        C, M = w.customers, w.merchants
        ts, T = cfg.test_start_sec, self.T
        rate_med = float(np.median(C["rate"][: self.n_cust0]))
        cust = -1
        for _ in range(100):
            c = int(rng.integers(0, self.n_cust0))
            if (C["open_t"][c] < ts - 120 * DAY and C["rate"][c] > rate_med and not C["is_mule"][c]
                    and c not in self.bust_out_end):
                cust = c
                break
        if cust < 0:
            return 0
        t0 = rng.uniform(ts, max(T - 4 * DAY, ts + HOUR))
        dur = rng.uniform(1, 4) * DAY
        n = int(rng.integers(8, 31))
        t = (t0 + np.sort(rng.random(n)) * dur).astype(np.int64)
        t = t[t < T]
        n = len(t)
        if n == 0:
            return 0
        names = ["electronics", "jewelry", "gift_cards", "department_store", "airlines", "crypto_quasi_cash"]
        mcc = rng.choice([self.mcc_by_name[x] for x in names], n, p=[0.35, 0.15, 0.15, 0.15, 0.1, 0.1])
        cnp = (rng.random(n) < 0.5) | (w.mccs["cnp_share"].to_numpy()[mcc] >= 0.9)
        merchant, cnp = w.picker.pick(rng, mcc, np.full(n, C["city"][cust]), cnp)
        cnp = cnp | (M["mode"][merchant] == "cnp")
        terminal = pick_terminal(rng, w, merchant, ~cnp)
        mu, sg = self._mu_sigma(merchant)
        value = np.exp(mu + 1.0 + C["spend_offset"][cust] + 0.8 * sg * rng.normal(0, 1, n))
        cust_arr = np.full(n, cust)
        device = np.where(cnp, current_device(rng, w, cust_arr, t, p_latest=1.0), -1)
        ip = np.where(cnp, home_ip(w, cust_arr, t), -1)
        card = C["card_start"][cust] + rng.integers(0, C["n_cards"][cust], n)
        mcc_m = M["mcc_idx"][merchant]
        phys = (mcc_m >= 0) & w.mccs["physical"].to_numpy()[mcc_m.clip(0)] & cnp
        iid = self._new_incident("bust_out", start=int(t[0]), end=int(t[-1]), customer=cust)
        kept = self.rows.add(n, t=t, customer=cust, card=card, merchant=merchant, terminal=terminal,
                      channel=np.where(cnp, "CNP", "CP").astype(object),
                      pos_entry_mode=np.where(cnp, "ecom", rng.choice(["chip", "contactless"], n, p=[0.7, 0.3])).astype(object),
                      value_usd=value, device=device, ip=ip,
                      email_domain=np.where(cnp, C["email"][cust], None).astype(object),
                      shipping_country=np.where(phys, C["country"][cust], -1),
                      cred_type=np.where(cnp, "owner", "card_present").astype(object),
                      is_fraud=1, scenario="bust_out", incident=iid, delay_days=rng.uniform(30, 90, n))
        self.bust_out_end[cust] = int(t[-1])
        return kept

    # ── driver ──
    def run(self, n_legit: int) -> None:
        cfg = self.cfg
        r = cfg.target_fraud_rate
        total = r / (1 - r) * n_legit
        test_frac = cfg.test_days / cfg.n_days
        novel = cfg.novel_share_of_test_fraud / (1 - cfg.novel_share_of_test_fraud) * total * test_frac
        rest = total - novel
        shares = {k: v for k, v in cfg.scenario_shares.items() if k != "friendly_fraud"}
        norm = sum(cfg.scenario_shares.values())
        self.friendly_budget = int(rest * cfg.scenario_shares.get("friendly_fraud", 0) / norm)
        order = ["mule_ring", "card_testing", "stolen_card_cnp", "account_takeover", "terminal_cashout", "skimming"]
        self.budgets = {k: int(rest * shares.get(k, 0) / norm) for k in order}
        self.budgets["bust_out"] = int(novel)
        for name in order + ["bust_out"]:
            budget = self.budgets[name]
            fn = getattr(self, name)
            made, tries = 0, 0
            while made < budget and tries < 200_000:
                made += fn()
                tries += 1

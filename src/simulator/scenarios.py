"""
Fraud processes. Each scenario is an attacker agent that *generates* transactions;
``is_fraud`` (= ground truth ``is_fraud_true``) / ``fraud_scenario`` record which process produced a row.

Incidents are SCHEDULED day by day (``FraudFactory.step``) with campaign waves and a drifting
mix, and their rows are queued until their day arrives (``take``). The closed loop
(loop.py) reports every decision back through ``feedback`` so that gangs adapt.

Scenarios (ground truth in ``fraud_scenario``):
  card_testing      bursts of tiny CNP charges on one merchant; stolen-list or BIN-enumeration
                    attacks (invalid PANs -> 14/54 declines, owned by phantom customers); cards
                    that pass are added to the compromised pool for later cash-out.
  stolen_card_cnp   compromised card used online by a gang: gang devices / IPs / emails reused
                    across incidents, resale-friendly merchants, reshipping; tactics adapt.
  account_takeover  failed logins -> login from a new device -> credential change -> dwell ->
                    purchases with the stored card and P2P pushes to mules (events stream).
  terminal_cashout  a crew runs counterfeit / stolen cards through one terminal for 1-4 weeks.
  skimming          cards used at a skimmed terminal are cloned and used card-present elsewhere.
  mule_ring         AMLworld-style laundering through chains / scatter-gather / cycles.
  synthetic_identity  fake identities build thin, good credit behaviour for months, then spree.
  merchant_collusion  new merchant ramps volume with stolen + complicit cards, refund abuse,
                    chargeback spike, disappears.
  transaction_laundering  a legitimate-looking front merchant processes stolen cards for others.
  bust_out          NOVEL scenario, test period only: long-standing accounts max out on
                    high-value goods from their usual device and IP (held out of training).
  friendly_fraud    (loop.py) legit purchases later disputed: ``is_fraud_true`` = 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import geo, seasonality
from .config import NOVEL_SCENARIO, SCENARIOS
from .entities import NEG_INF, World, _weighted_city_global
from .legit import current_device, home_ip, pick_terminal
from .rows import Rows

HOUR, DAY = 3600, 86_400
AMT_MULT = np.array([0.45, 1.0, 1.8])
ORDER = ["mule_ring", "card_testing", "stolen_card_cnp", "account_takeover", "terminal_cashout", "skimming",
         "synthetic_identity", "merchant_collusion", "transaction_laundering", NOVEL_SCENARIO]
AVG_ROWS = {"mule_ring": 60, "card_testing": 100, "stolen_card_cnp": 4, "account_takeover": 4,
            "terminal_cashout": 80, "skimming": 25, "synthetic_identity": 12, "merchant_collusion": 120,
            "transaction_laundering": 60, NOVEL_SCENARIO: 15}


class Tactics:
    """Per-gang factored bandit over tactic arms; rewards come from the closed loop's decisions.

    dims: 0 ip kind (hosting / proxy / home), 1 amount band (low / mid / high),
          2 merchant group (3 groups of the gang's merchant list), 3 card age (fresh / aged)."""
    DIMS = {"ip": 0, "amt": 1, "mg": 2, "aged": 3}
    PRIOR = np.array([[0.35, 0.50, 0.15], [0.25, 0.50, 0.25], [1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0.0]])

    def __init__(self, n_gangs: int, adaptive: bool, rng: np.random.Generator):
        self.adaptive, self.rng = adaptive, rng
        self.succ = np.zeros((n_gangs, 4, 3))
        self.n = np.zeros((n_gangs, 4, 3))

    def choose(self, g: int, dim: str) -> int:
        d = self.DIMS[dim]
        p = self.PRIOR[d]
        if not self.adaptive or self.rng.random() < 0.08:
            return int(self.rng.choice(3, p=p))
        a = 1.0 + 2.0 * p + self.succ[g, d]
        b = 1.0 + 2.0 * (1 - p) + (self.n[g, d] - self.succ[g, d])
        x = self.rng.beta(a, b)
        x[p == 0] = -1.0
        return int(np.argmax(x))

    def update(self, gang: np.ndarray, arms: np.ndarray, reward: np.ndarray) -> None:
        for d in range(4):
            m = arms[:, d] >= 0
            if m.any():
                np.add.at(self.succ, (gang[m], d, arms[m, d]), reward[m])
                np.add.at(self.n, (gang[m], d, arms[m, d]), 1.0)

    def decay(self, f: float = 0.985) -> None:
        self.succ *= f
        self.n *= f


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
        self.events: list[dict] = []                 # account events produced by attackers
        self.pool: list[tuple[int, int]] = []        # (card, time compromised)
        self.n_cust0 = self.cfg.n_customers
        self.n_card0 = len(world.cards)
        self.next_address = int(world.customers["address"].max()) + 1
        self.legit = legit
        self._parts_seen = 0
        self._buf: dict | None = None

        M, mccs = world.merchants, world.mccs
        name_of = dict(enumerate(mccs["name"]))
        self.mcc_by_name = {v: k for k, v in name_of.items()}
        mcc = M["mcc_idx"]
        mode = M["mode"]
        high = np.zeros(len(mcc), bool)
        high[mcc >= 0] = mccs["high_risk"].to_numpy()[mcc[mcc >= 0]]
        self.online = np.nonzero(np.isin(mode, ["cnp", "omni"]) & (mcc >= 0))[0]
        self.online_high = np.nonzero(np.isin(mode, ["cnp", "omni"]) & high)[0]
        low_names = [self.mcc_by_name[n] for n in ("streaming", "misc_retail", "clothing", "donations")]
        self.online_low = np.nonzero(np.isin(mode, ["cnp", "omni"]) & np.isin(mcc, low_names))[0]
        m_country = world.cities["country_idx"].to_numpy()[M["city"]]
        self.m_country = m_country
        # gangs favour cross-border stores (weaker issuer checks, easier resale)
        w_high = np.where(m_country[self.online_high] != 0, 6.0, 1.0)
        self.online_high_p = w_high / w_high.sum()
        pop = M["popularity"][self.online]
        self.online_p = pop / pop.sum()                  # ordinary online traffic mix (popularity weighted)
        sleeper_mccs = [self.mcc_by_name[n] for n in ("digital_goods", "games", "electronics", "gift_cards", "misc_retail")]
        self.sleepers = np.nonzero(np.isin(mode, ["cnp", "omni"]) & np.isin(mcc, sleeper_mccs))[0]
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
        self.tactics = Tactics(self.cfg.n_gangs, self.cfg.adaptive_attackers, rng)
        self.dead: set[int] = set()                  # (incident, card) pairs the attacker abandoned
        self._plan_budgets(len(legit["t"]))

    # ── budgets, daily intensities ──
    def _plan_budgets(self, n_legit: int) -> None:
        cfg, rng = self.cfg, self.rng
        r = cfg.target_fraud_rate
        total = r / (1 - r) * n_legit
        test_frac = cfg.test_days / cfg.n_days
        novel = cfg.novel_share_of_test_fraud / (1 - cfg.novel_share_of_test_fraud) * total * test_frac
        rest = total - novel
        shares = {k: cfg.scenario_shares.get(k, 0.0) for k in SCENARIOS}
        norm = sum(shares.values())
        self.budgets = {k: float(rest * shares[k] / norm) for k in SCENARIOS}
        self.budgets[NOVEL_SCENARIO] = float(novel)
        n = cfg.n_days
        legit_w = self.w.cal_w
        fw = seasonality.fraud_day_weights(rng, legit_w, cfg.campaign_waves)
        self.fraud_w = fw
        self.cumw = {}
        for s in self.budgets:
            wd = fw * seasonality.mix_weights(n, s, cfg.mix_shift)
            if s != NOVEL_SCENARIO:
                # stratify: every period (train / val / test) receives its volume-proportional share of the
                # scenario budget; campaign waves and the mix drift still shape the days inside a period
                bounds = [0, cfg.val_start_sec // DAY, cfg.test_start_sec // DAY, n]
                tot = wd.sum()
                for a, b in zip(bounds[:-1], bounds[1:]):
                    if b > a and wd[a:b].sum() > 0:
                        wd[a:b] *= tot * legit_w[a:b].sum() / legit_w.sum() / wd[a:b].sum()
            if s == NOVEL_SCENARIO:
                wd = np.where(np.arange(n) * DAY >= cfg.test_start_sec, wd, 0.0)
            self.cumw[s] = np.cumsum(wd) / max(wd.sum(), 1e-9)
        self.made = {s: 0.0 for s in self.budgets}
        self.period_made = {(s, p): 0.0 for s in self.budgets for p in range(3)}
        self.avg_rows = dict(AVG_ROWS)

    def step(self, day: int) -> None:
        """Create the incidents that start on ``day`` (deficit-driven towards the cumulative target)."""
        rng = self.rng
        for s in ORDER:
            if self.budgets.get(s, 0) <= 0:
                continue
            target = self.budgets[s] * self.cumw[s][day]
            guard = 0
            while guard < 300:
                deficit = target - self.made[s]
                # stochastic rounding keeps the expected total on budget even when one incident is
                # larger than the whole budget (merchant collusion, terminal cash-outs, ...)
                if deficit <= 0 or rng.random() > min(1.0, deficit / max(self.avg_rows[s], 1.0)):
                    break
                guard += 1
                kept = getattr(self, s)((day + rng.random()) * DAY)
                if kept:
                    self.made[s] += kept
                    self.period_made[(s, self._period(day))] += kept
                    self.avg_rows[s] = 0.9 * self.avg_rows[s] + 0.1 * kept
        self._ensure_coverage(day)

    def _period(self, day: int) -> int:
        return int(day >= self.cfg.val_start_sec // DAY) + int(day >= self.cfg.test_start_sec // DAY)

    def _ensure_coverage(self, day: int) -> None:
        """Lumpy scenarios (few, large incidents) must not skip a whole period: at the middle of each
        period, a scenario whose expected share there is at least ~half an average incident but which has
        produced nothing yet is forced to launch one incident (a campaign scheduled later in the period)."""
        cfg, rng = self.cfg, self.rng
        bounds = [0, cfg.val_start_sec // DAY, cfg.test_start_sec // DAY, cfg.n_days]
        pi = self._period(day)
        a, b = bounds[pi], bounds[pi + 1]
        if day != a + (b - a) // 2:
            return
        for s in ORDER:
            if s == NOVEL_SCENARIO or self.budgets.get(s, 0) <= 0 or self.period_made[(s, pi)] > 0:
                continue
            share = self.w.cal_w[a:b].sum() / self.w.cal_w.sum()
            if self.budgets[s] * share >= 0.5 * self.avg_rows[s]:
                kept = getattr(self, s)((day + rng.random()) * DAY)
                if kept:
                    self.made[s] += kept
                    self.period_made[(s, pi)] += kept

    def preroll(self, days: int = 21) -> None:
        """Attacks that started before day 0 and are still running (rows with t < 0 are dropped)."""
        rng = self.rng
        for d in range(-days, 0):
            for s in ORDER:
                if s == NOVEL_SCENARIO or self.budgets.get(s, 0) <= 0:
                    continue
                lam = self.budgets[s] / self.cfg.n_days / max(self.avg_rows[s], 1.0)
                for _ in range(rng.poisson(lam)):
                    getattr(self, s)((d + rng.random()) * DAY)

    def take(self, day_end: int) -> dict | None:
        """Pop all queued rows with t < day_end (seconds) as a dict of arrays."""
        parts = self.rows.parts[self._parts_seen:]
        self._parts_seen = len(self.rows.parts)
        if parts:
            new = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            self._buf = new if self._buf is None else {k: np.concatenate([self._buf[k], new[k]]) for k in new}
        if self._buf is None or not len(self._buf["t"]):
            return None
        m = self._buf["t"] < day_end
        if not m.any():
            return None
        out = {k: v[m] for k, v in self._buf.items()}
        self._buf = {k: v[~m] for k, v in self._buf.items()}
        return out

    def abandon(self, incident: np.ndarray, card: np.ndarray) -> None:
        """Attacker gives up on (incident, card) after a decline: queued rows are cancelled."""
        for i, c in zip(incident.tolist(), card.tolist()):
            self.dead.add(int(i) * 100_000_000 + int(c))
        if self._buf is not None and self.dead and len(self._buf["t"]):
            key = self._buf["incident"] * 100_000_000 + self._buf["card"]
            keep = ~np.isin(key, np.fromiter(self.dead, dtype=np.int64, count=len(self.dead)))
            if not keep.all():
                self._buf = {k: v[keep] for k, v in self._buf.items()}

    def feedback(self, gang, tac_ip, tac_amt, tac_mg, tac_aged, approved) -> None:
        if len(gang) == 0:
            return
        arms = np.stack([tac_ip, tac_amt, tac_mg, tac_aged], axis=1).astype(np.int64)
        mult = np.where(arms[:, 1] >= 0, AMT_MULT[arms[:, 1].clip(0)], 1.0)
        reward = approved.astype(float) * np.sqrt(mult) / np.sqrt(AMT_MULT.max())
        self.tactics.update(gang.astype(np.int64), arms, reward)

    def end_of_day(self) -> None:
        self.tactics.decay()

    # ── helpers ──
    def _new_incident(self, scenario: str, **info) -> int:
        iid = len(self.incidents)
        self.incidents.append({"incident": iid, "scenario": scenario, **info})
        return iid

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
            shared = rng.choice(w.vpn_pool, (n_h * 3) // 4 + 1)     # datacenter exits shared with legit VPN users
            own_h = w.new_ips(rng, np.full(n_h - len(shared) if n_h > len(shared) else 1, gc), w.city_of_country(rng, np.full(n_h - len(shared) if n_h > len(shared) else 1, gc)), "hosting")
            n_u = int(rng.integers(2, 8))
            host_us = w.new_ips(rng, np.zeros(n_u, np.int64), rng.choice(us_cities, n_u), "hosting")
            n_p = int(rng.integers(3, 16))
            proxy = w.new_ips(rng, np.zeros(n_p, np.int64), rng.choice(us_cities, n_p), "residential")
            n_r = int(rng.integers(2, 9))
            home = w.new_ips(rng, np.full(n_r, gc), w.city_of_country(rng, np.full(n_r, gc)), "residential")
            self.gangs.append({
                "country": gc, "devices": devices,
                "ips": {"hosting": np.concatenate([shared, own_h]), "hosting_us": host_us, "proxy": proxy, "home": home},
                "emails": rng.choice(geo.FRAUD_EMAIL[0], 3, p=geo.FRAUD_EMAIL[1]),
                "merchants": rng.choice(self.online_high, int(rng.integers(10, 41)), p=self.online_high_p),
            })
        wts = rng.pareto(1.2, self.cfg.n_gangs) + 1
        self.gang_p = wts / wts.sum()

    def _email(self, gang) -> str:
        """Half of the time a fresh throwaway address from the global fraud mix, else the gang's habit."""
        if self.rng.random() < 0.75:
            return str(self.rng.choice(geo.FRAUD_EMAIL[0], p=geo.FRAUD_EMAIL[1]))
        return str(self.rng.choice(gang["emails"]))

    def _legit_email(self) -> str:
        return str(self.rng.choice(geo.LEGIT_EMAIL[0], p=np.asarray(geo.LEGIT_EMAIL[1]) / sum(geo.LEGIT_EMAIL[1])))

    def _gang(self):
        g = int(self.rng.choice(len(self.gangs), p=self.gang_p))
        return g, self.gangs[g]

    def _ips_for_arm(self, gang, arm: int, n: int, victim: int = -1, clean: bool = False):
        """IPs for n rows when the gang's tactic is ``arm`` (0 hosting, 1 proxy, 2 home)."""
        rng = self.rng
        arms = np.where(rng.random(n) < 0.65, arm, rng.integers(0, 3, n))
        if clean:
            arms = np.ones(n, dtype=np.int64)
        out = np.empty(n, dtype=np.int64)
        ips = gang["ips"]
        m = arms == 0
        us_dc = rng.random(m.sum()) < 0.8                # datacenter in the victim's (mostly US) region
        out[m] = np.where(us_dc, rng.choice(ips["hosting_us"], m.sum()), rng.choice(ips["hosting"], m.sum()))
        m = arms == 1
        k = int(m.sum())
        if k:
            sel = rng.choice(ips["proxy"], k)
            if victim >= 0:     # residential proxy exiting in the victim's own city
                C = self.w.customers
                near = (rng.random(k) < 0.7) | clean
                if near.any():
                    sel[near] = self.w.new_ips(rng, np.full(near.sum(), C["country"][victim]),
                                               np.full(near.sum(), C["city"][victim]), "residential")
            out[m] = sel
        m = arms == 2
        out[m] = rng.choice(ips["home"], m.sum())
        return out, arms.astype(np.int8)

    def _cust_ok(self, cust, t):
        C = self.w.customers
        return (cust >= 0) and (C["open_t"][cust] < t)

    def _card_ok(self, card, t) -> bool:
        bt = self.w.cards["blocked_t"][card]
        return not (bt < t)

    def _random_customer(self, t, exclude_mules=True):
        C = self.w.customers
        for _ in range(50):
            c = int(self.rng.integers(0, self.n_cust0))
            if self._cust_ok(c, t) and not (exclude_mules and C["is_mule"][c]):
                return c
        return int(self.rng.integers(0, self.n_cust0))

    def _pool_card(self, t0: float, aged_days: float = 0.5):
        """A compromised card that is at least ``aged_days`` old, or None."""
        rng = self.rng
        for _ in range(8):
            if not self.pool:
                return None
            i = int(rng.integers(0, len(self.pool)))
            c, tc = self.pool[i]
            if tc + aged_days * DAY <= t0 and self._card_ok(c, t0):
                self.pool.pop(i)
                return c
        return None

    def _any_card(self, t0: float):
        c = self._pool_card(t0)
        return c if c is not None else self._random_card(t0, 5)

    def _random_card(self, t0: float, tries: int = 20):
        cards = self.w.cards
        for _ in range(tries):
            c = int(self.rng.integers(0, self.n_card0))
            if self._card_ok(c, t0) and self._cust_ok(int(cards["owner"][c]), max(t0, 0)):
                return c
        return None

    def _event(self, t, customer, kind, device=-1, ip=-1, success=True, incident=-1, email=None, payee=-1):
        self.events.append(dict(t=int(t), customer=int(customer), type=kind, device=int(device), ip=int(ip),
                                success=bool(success), incident=int(incident), email_domain=email,
                                counterparty=int(payee)))

    # ── scenarios: each makes ONE incident starting at t0 and returns the number of rows kept ──
    def card_testing(self, t0: float) -> int:
        w, rng = self.w, self.rng
        cards, C = w.cards, w.customers
        g, gang = self._gang()
        m = int(rng.choice(self.test_merchants))
        dur = float(np.exp(rng.uniform(np.log(600), np.log(6 * HOUR))))
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
            f_country = np.full(n_fake, cards["country"][ref])
            f_net = np.full(n_fake, cards["network"][ref], dtype=object)
            f_typ = np.full(n_fake, cards["card_type"][ref], dtype=object)
            f_bin = np.full(n_fake, b)
            valid_cred = np.full(k_valid, "enumerated_valid", dtype=object)
        else:
            k_valid = rng.binomial(n_cards, rng.uniform(0.15, 0.5))
            valid = rng.choice(self.n_card0, k_valid, replace=False)
            n_fake = n_cards - k_valid
            f_country = rng.choice(len(w.countries), n_fake, p=w.countries["customer_w"].to_numpy())
            f_net = rng.choice(["visa", "mastercard", "amex", "discover"], n_fake, p=[0.55, 0.3, 0.1, 0.05]).astype(object)
            f_typ = rng.choice(["debit", "credit", "prepaid"], n_fake, p=[0.55, 0.35, 0.10]).astype(object)
            f_bin = w.random_bins(rng, f_country, f_net, f_typ)
            valid_cred = rng.choice(["stolen_full", "stolen_partial"], k_valid, p=[0.5, 0.5]).astype(object)
        # invented PANs are owned by phantom customers so that "no customer" is not a fraud flag
        f_city = w.city_of_country(rng, f_country) if n_fake else np.zeros(0, np.int64)
        phantom = C.append(country=f_country, city=f_city, tz=w.cities["tz"].to_numpy()[f_city].astype(float),
                           open_t=NEG_INF, kind="phantom", n_cards=1) if n_fake else np.zeros(0, np.int64)
        fake = cards.append(owner=phantom, country=f_country, network=f_net, card_type=f_typ, bin=f_bin,
                            issued_t=NEG_INF) if n_fake else np.zeros(0, np.int64)
        card_list = np.concatenate([valid, fake]).astype(np.int64)
        cred_list = np.concatenate([valid_cred, np.full(n_fake, "invalid_pan", dtype=object)])
        att = np.concatenate([np.arange(n_cards), rng.integers(0, n_cards, max(n_att - n_cards, 0))])
        rng.shuffle(att)
        n = len(att)
        t = (t0 + np.sort(rng.random(n)) * dur).astype(np.int64)
        card = card_list[att]
        owner = cards["owner"][card]
        devs = rng.choice(gang["devices"], int(rng.integers(1, 3)))
        device = rng.choice(devs, n) if rng.random() < 0.8 else np.full(n, -1)
        value = np.where(rng.random(n) < 0.3, 1.0, rng.uniform(0.5, 5.0, n))
        arm = self.tactics.choose(g, "ip")
        ip, ip_arm = self._ips_for_arm(gang, arm, n)
        iid = self._new_incident("card_testing", gang=g, start=int(t[0]), end=int(t[-1]), merchant=m,
                                 enumerated=bool(enumerated), n_cards=int(n_cards), n_valid=int(len(valid)))
        kept = self.rows.add(n, t=t, customer=owner, card=card, merchant=m, terminal=w.merchants["term_start"][m],
                             channel="CNP", pos_entry_mode="ecom", value_usd=value, device=device, ip=ip,
                             email_domain=rng.choice(geo.FRAUD_EMAIL[0], n, p=geo.FRAUD_EMAIL[1]).astype(object),
                             cred_type=cred_list[att], is_fraud=1, scenario="card_testing", incident=iid,
                             gang=g, tac_ip=ip_arm)
        last_t = pd.Series(t).groupby(card).max()
        for c in valid:
            self.pool.append((int(c), int(last_t.get(c, t0))))
        return kept

    def stolen_card_cnp(self, t0: float) -> int:
        w, rng = self.w, self.rng
        cards, M = w.cards, w.merchants
        g, gang = self._gang()
        aged = self.tactics.choose(g, "aged") if rng.random() < 0.9 else 0
        card = self._pool_card(t0) if (aged == 1 or rng.random() < 0.3) else None
        if card is None:
            card = self._random_card(t0)
        if card is None:
            return 0
        owner = int(cards["owner"][card])
        n = int(min(1 + rng.geometric(0.35), 10))
        t = (t0 + np.sort(rng.random(n)) * rng.uniform(2 * HOUR, 4 * DAY)).astype(np.int64)
        mg = self.tactics.choose(g, "mg")
        groups = np.array_split(gang["merchants"], 3)
        gm = groups[mg] if len(groups[mg]) else gang["merchants"]
        clean = rng.random() < self.cfg.clean_attack_share   # a careful attacker who blends in with the victim
        merchant = np.where((rng.random(n) < 0.2) & (not clean), rng.choice(gm, n), rng.choice(self.online, n, p=self.online_p))
        amt = self.tactics.choose(g, "amt")
        mu, sg = self._mu_sigma(merchant)
        value = np.exp(mu + (0.0 if clean else 0.1) + sg * rng.normal(0, 1, n)) * (1.0 if clean else AMT_MULT[amt])
        if rng.random() < 0.3:
            value[0] = rng.uniform(1, 10)
        dev = int(rng.choice(gang["devices"])) if rng.random() < 0.5 else int(w.new_devices(rng, 1, -1, "gang", True)[0])
        if clean:
            dev = int(w.new_devices(rng, 1, -1, "gang", False)[0])
        if rng.random() < 0.10:
            dev = -1                                         # fingerprint blocked / not collected
        arm = self.tactics.choose(g, "ip")
        ip, ip_arm = self._ips_for_arm(gang, arm, n, victim=owner, clean=clean)
        mcc = M["mcc_idx"][merchant]
        phys = w.mccs["physical"].to_numpy()[mcc.clip(0)] & (mcc >= 0)
        ship = np.where(phys, np.where((rng.random(n) < 0.80) | clean, cards["country"][card], gang["country"]), -1)
        cred = "owner" if clean else ("stolen_full" if rng.random() < 0.8 else "stolen_partial")   # a phished OTP-grade kit looks like the owner
        iid = self._new_incident("stolen_card_cnp", gang=g, start=int(t[0]), end=int(t[-1]), card=card)
        return self.rows.add(n, t=t, customer=owner, card=card, merchant=merchant,
                             terminal=M["term_start"][merchant], channel="CNP", pos_entry_mode="ecom",
                             value_usd=value, device=dev, ip=ip, email_domain=self._legit_email() if clean else self._email(gang),
                             shipping_country=ship, cred_type=cred, is_fraud=1, scenario="stolen_card_cnp",
                             incident=iid, gang=g, tac_ip=ip_arm, tac_amt=amt, tac_mg=mg, tac_aged=aged,
                             guest=int(rng.random() < 0.20))

    def account_takeover(self, t0: float) -> int:
        w, rng = self.w, self.rng
        C, M = w.customers, w.merchants
        g, gang = self._gang()
        victim = -1
        for _ in range(50):
            c = int(rng.integers(0, self.n_cust0))
            if C["online"][c] > 1.0 and self._cust_ok(c, max(t0, 0)) and not C["is_mule"][c]:
                victim = c
                break
        if victim < 0:
            return 0
        card = int(C["card_start"][victim])
        clean = rng.random() < self.cfg.clean_attack_share
        dev = int(rng.choice(gang["devices"])) if rng.random() < 0.4 else int(w.new_devices(rng, 1, -1, "gang", True)[0])
        if clean:
            dev = int(w.new_devices(rng, 1, -1, "gang", False)[0])
        arm = self.tactics.choose(g, "ip")
        amt = self.tactics.choose(g, "amt") if not clean else 1
        email = self._email(gang) if not clean else self._legit_email()
        n_p = int(rng.integers(1, 6))
        n_x = int(rng.integers(0, 4)) if self.rings else 0
        n = n_p + n_x
        dwell = rng.uniform(0.3 * HOUR, 30 * HOUR)
        t = (t0 + dwell + np.sort(rng.random(n)) * rng.uniform(2 * HOUR, 2 * DAY)).astype(np.int64)
        ip, ip_arm = self._ips_for_arm(gang, arm, n, victim=victim, clean=clean)
        is_x = np.zeros(n, bool)
        is_x[rng.choice(n, n_x, replace=False)] = True
        merchant = np.where(is_x, w.picker.p2p[victim % len(w.picker.p2p)],
                            np.where(rng.random(n) < 0.25, rng.choice(self.online_high, n, p=self.online_high_p),
                                     rng.choice(self.online, n, p=self.online_p)))
        mu, sg = self._mu_sigma(merchant)
        value = np.where(is_x, rng.lognormal(5.4, 0.8, n), np.exp(mu + 0.12 + sg * rng.normal(0, 1, n)) * AMT_MULT[amt])
        counterparty = np.full(n, -1, dtype=np.int64)
        ring = np.full(n, -1, dtype=np.int64)
        if n_x:
            active = [r for r in self.rings if r["start"] <= t0 <= r["end"]] or self.rings
            r = active[int(rng.integers(0, len(active)))]
            counterparty[is_x] = rng.choice(r["entries"], n_x)
            ring[is_x] = r["ring"]
        mcc = M["mcc_idx"][merchant]
        phys = (mcc >= 0) & w.mccs["physical"].to_numpy()[mcc.clip(0)]
        ship = np.where(phys, np.where((rng.random(n) < 0.8) | clean, C["country"][victim], gang["country"]), -1)
        iid = self._new_incident("account_takeover", gang=g, start=int(t0), end=int(t[-1]), customer=victim)
        # account-event trail: failed logins -> login from a new device -> credential change -> dwell
        login_ip = int(ip[0])
        for k in range(int(rng.integers(0, 6))):
            self._event(t0 - rng.uniform(30, 1800), victim, "login", dev, int(rng.choice(gang["ips"]["hosting"])),
                        False, iid)
        self._event(t0, victim, "login", dev, login_ip, True, iid)
        u = rng.random()
        if u < 0.55:
            self._event(t0 + rng.uniform(60, 1200), victim, "password_reset", dev, login_ip, True, iid)
        elif u < 0.80:
            self._event(t0 + rng.uniform(60, 1200), victim, "email_change", dev, login_ip, True, iid, email=email)
        else:
            self._event(t0 + rng.uniform(60, 1200), victim, "phone_change", dev, login_ip, True, iid)
        if rng.random() < 0.4:
            self._event(t0 + rng.uniform(60, 600), victim, "device_enrolment", dev, login_ip, True, iid)
        for tx_t, cp in zip(t[is_x], counterparty[is_x]):
            self._event(tx_t - rng.uniform(60, 600), victim, "new_payee", dev, login_ip, True, iid, payee=cp)
        for c in rng.integers(0, self.n_cust0, int(rng.integers(8, 30))):      # credential-stuffing noise
            self._event(t0 - rng.uniform(0, 1800), int(c), "login", dev, int(rng.choice(gang["ips"]["hosting"])), False, iid)
        return self.rows.add(n, t=t, customer=victim, card=card, merchant=merchant, terminal=M["term_start"][merchant],
                             counterparty=counterparty, channel=np.where(is_x, "P2P", "CNP").astype(object),
                             pos_entry_mode=np.where(is_x, "p2p", np.where(rng.random(n) < 0.5, "ecom", "card_on_file")).astype(object),
                             value_usd=value, device=dev, ip=ip, email_domain=email, shipping_country=ship,
                             cred_type=np.where(is_x, "account_login", "card_on_file").astype(object),
                             is_fraud=1, scenario="account_takeover", incident=iid, ring=ring,
                             gang=g, tac_ip=ip_arm, tac_amt=amt)

    def terminal_cashout(self, t0: float) -> int:
        w, rng = self.w, self.rng
        cards = w.cards
        if not len(self.cashout_terminals):
            return 0
        term = int(rng.choice(self.cashout_terminals))
        m = int(w.terminals["merchant"][term])
        W = rng.uniform(7, 28) * DAY
        n = int(rng.poisson(rng.uniform(2, 6) * W / DAY))
        if n == 0:
            return 0
        n_cards = max(1, int(n / 1.5))
        got = [self._pool_card(t0) for _ in range(n_cards // 2)]
        pool_cards = [c for c in got if c is not None]
        extra = [c for c in (self._random_card(t0, 5) for _ in range(n_cards - len(pool_cards))) if c is not None]
        uniq = np.array(pool_cards + extra, dtype=np.int64)
        if not len(uniq):
            return 0
        card = np.concatenate([uniq, rng.choice(uniq, max(n - len(uniq), 0))])[:n]
        t = (t0 + np.sort(rng.random(n)) * W).astype(np.int64)
        mu, sg = self._mu_sigma(np.full(n, m))
        iid = self._new_incident("terminal_cashout", start=int(t[0]), end=int(t[-1]), terminal=term, merchant=m)
        return self.rows.add(n, t=t, customer=cards["owner"][card], card=card, merchant=m, terminal=term, channel="CP",
                             pos_entry_mode=rng.choice(["swipe", "keyed", "chip", "contactless"], n,
                                                       p=[0.30, 0.12, 0.36, 0.22]).astype(object),
                             value_usd=np.exp(mu + 0.12 + sg * rng.normal(0, 1, n)), cred_type="counterfeit",
                             is_fraud=1, scenario="terminal_cashout", incident=iid)

    def skimming(self, t0: float) -> int:
        w, rng = self.w, self.rng
        cards, cities = w.cards, w.cities
        if not len(self.skim_terminals):
            return 0
        term = int(rng.choice(self.skim_terminals))
        W = rng.uniform(5, 30) * DAY
        s0 = t0 - W - rng.uniform(0, 20) * DAY          # the skimmer is read BEFORE the crew acts
        big = np.int64(1_000_000_000)
        lo = np.searchsorted(self.leg_key, term * big + max(int(s0), 0))
        hi = np.searchsorted(self.leg_key, term * big + max(int(s0 + W), 0))
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
        t = np.repeat(c_t + rng.uniform(2, 30, k) * DAY, n_use) + rng.uniform(0, 2 * DAY, len(card))
        t = np.where(t < t0, t0 + rng.uniform(0, 3 * DAY, len(t)), t).astype(np.int64)
        n = len(card)
        names = ["electronics", "jewelry", "department_store", "atm_cash", "gift_cards", "grocery", "fuel",
                 "restaurants", "misc_retail"]
        mcc = rng.choice([self.mcc_by_name[x] for x in names], n, p=[0.15, 0.06, 0.12, 0.17, 0.03, 0.20, 0.12, 0.08, 0.07])
        merchant, online = w.picker.pick(rng, mcc, np.full(n, crew_city), np.zeros(n, bool))
        keep = ~online  # counterfeit stripes only work card-present
        keep &= np.array([self._card_ok(int(c), tt) for c, tt in zip(card, t)], dtype=bool) if n else keep
        card, t, merchant = card[keep], t[keep], merchant[keep]
        n = len(card)
        if n == 0:
            return 0
        terminal = pick_terminal(rng, w, merchant, np.ones(n, bool))
        mu, sg = self._mu_sigma(merchant)
        iid = self._new_incident("skimming", start=int(t.min()), end=int(t.max()), terminal=term,
                                 skim_start=int(max(s0, 0)), skim_end=int(max(s0 + W, 0)), n_cards=int(k))
        return self.rows.add(n, t=t, customer=cards["owner"][card], card=card, merchant=merchant, terminal=terminal,
                             channel="CP", pos_entry_mode=rng.choice(["swipe", "keyed", "chip", "contactless"], n,
                                                                      p=[0.35, 0.10, 0.33, 0.22]).astype(object),
                             value_usd=np.exp(mu + 0.12 + sg * rng.normal(0, 1, n)), cred_type="counterfeit",
                             is_fraud=1, scenario="skimming", incident=iid)

    def mule_ring(self, t0: float) -> int:
        w, rng = self.w, self.rng
        C, cities = w.customers, w.cities
        m = int(rng.integers(3, 13))
        dur = rng.uniform(3, 21) * DAY
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
                                                         np.full(n_new, "debit", dtype=object)),
                                       issued_t=NEG_INF)
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
                amt = float(rng.lognormal(5.0, 0.9))
                dev = int(current_device(rng, w, np.array([victim]), np.array([int(vt)]))[0]) if victim < self.n_cust0 else -1
                ip_ = int(home_ip(w, np.array([victim]), np.array([int(vt)]))[0])
                add(vt, victim, e, amt, dev, ip_, "mule_victim", p2p[victim % len(p2p)])
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
                    add(ts, src, dst, amt, rng.choice(r_dev), rng.choice(r_ip), "mule_layer", p2p[src % len(p2p)])
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
                        "mule_cashout", merch, "CNP" if cnp else "CP")
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
                             cred_type=cred, is_fraud=1, scenario="mule_ring", incident=iid, ring=ring_id, role=kind)
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

    def synthetic_identity(self, t0: float) -> int:
        """Credit-building ramp: months of thin, good behaviour (labelled legit: no loss yet),
        then a spree. Identities of one ring share an address / infrastructure."""
        w, rng, T = self.w, self.rng, self.T
        C, cities, cards, M = w.customers, w.cities, w.cards, w.merchants
        build = float(rng.uniform(45, 240))
        build = min(build, (T - 4 * DAY - t0) / DAY)
        if build < 21:
            return 0
        k = int(rng.integers(2, 6))
        gc = 0 if rng.random() < 0.8 else int(rng.choice(len(w.countries), p=w.countries["customer_w"].to_numpy()))
        city = w.city_of_country(rng, np.full(k, gc))
        ip = w.new_ips(rng, np.full(k, gc), city, "residential")
        shared_addr = rng.random() < 0.6
        addr0 = self.next_address
        self.next_address += 1 if shared_addr else k
        ids = C.append(country=gc, city=city, home_lat=cities["lat"].to_numpy()[city] + rng.normal(0, 0.08, k),
                       home_lon=cities["lon"].to_numpy()[city] + rng.normal(0, 0.08, k),
                       tz=cities["tz"].to_numpy()[city].astype(float), open_t=t0, rate=0.0, spend_offset=0.0,
                       online=1.0, p2p_share=0.0,
                       email=rng.choice(geo.LEGIT_EMAIL[0], k, p=geo.LEGIT_EMAIL[1]).astype(object),
                       address=np.full(k, addr0) if shared_addr else addr0 + np.arange(k), n_cards=1, n_dev=0,
                       home_start=ip, home_n=1, home_period=1e15, home_phase=0.0, kind="synthetic", is_mule=False)
        netw = rng.choice(["visa", "mastercard"], k).astype(object)
        c_ids = cards.append(owner=ids, country=gc, network=netw, card_type="credit",
                             bin=w.random_bins(rng, np.full(k, gc), netw, np.full(k, "credit", dtype=object)),
                             issued_t=t0)
        C["card_start"][ids] = c_ids
        share_dev = rng.random() < 0.3
        dev = w.new_devices(rng, 1 if share_dev else k, -1, "synthetic", fraud=False)
        dev = np.full(k, dev[0]) if share_dev else dev
        iid = self._new_incident("synthetic_identity", start=int(t0), build_days=build, n_identities=k,
                                 spree_start=int(t0 + build * DAY))
        everyday = ["grocery", "restaurants", "fuel", "pharmacy", "streaming", "telecom", "misc_retail"]
        ev_mcc = np.array([self.mcc_by_name[x] for x in everyday])
        # thin good behaviour: ~1-3 purchases a week, small amounts, local shops and a subscription
        n_b = rng.poisson(rng.uniform(1.0, 3.0, k) * build / 7)
        ident = np.repeat(np.arange(k), n_b)
        tb = (t0 + rng.uniform(1, build, len(ident)) * DAY).astype(np.int64)
        mcc_b = rng.choice(ev_mcc, len(ident))
        cnp_b = rng.random(len(ident)) < 0.35
        merch_b, cnp_b = w.picker.pick(rng, mcc_b, city[ident], cnp_b)
        mu, sg = self._mu_sigma(merch_b)
        val_b = np.exp(mu - 0.3 + 0.7 * sg * rng.normal(0, 1, len(ident)))
        cnp_b = cnp_b | (M["mode"][merch_b] == "cnp")
        dig = cnp_b
        ipb = np.where(dig, ip[ident], -1)
        self.rows.add(len(ident), t=tb, customer=ids[ident], card=c_ids[ident], merchant=merch_b,
                      terminal=pick_terminal(rng, w, merch_b, ~cnp_b),
                      channel=np.where(cnp_b, "CNP", "CP").astype(object),
                      pos_entry_mode=np.where(cnp_b, "ecom", "chip").astype(object), value_usd=val_b,
                      device=np.where(dig, dev[ident], -1), ip=ipb,
                      email_domain=np.where(dig, C["email"][ids[ident]], None).astype(object),
                      cred_type=np.where(cnp_b, "owner", "card_present").astype(object),
                      scenario="synthetic_identity_build", incident=iid, is_fraud=0)
        # the spree
        n_s = rng.integers(6, 20, k)
        ident = np.repeat(np.arange(k), n_s)
        ts = (t0 + build * DAY + rng.uniform(0, rng.uniform(1, 3), len(ident)) * DAY).astype(np.int64)
        names = ["electronics", "jewelry", "gift_cards", "department_store", "airlines", "atm_cash", "crypto_quasi_cash",
                 "clothing", "misc_retail", "hotels"]
        mcc_s = rng.choice([self.mcc_by_name[x] for x in names], len(ident),
                           p=[0.2, 0.08, 0.08, 0.12, 0.08, 0.08, 0.04, 0.14, 0.1, 0.08])
        mcc_s = np.where(rng.random(len(ident)) < 0.35, rng.choice(ev_mcc, len(ident)), mcc_s)    # part of the spree looks like everyday spending
        cnp_s = (rng.random(len(ident)) < 0.6) | (w.mccs["cnp_share"].to_numpy()[mcc_s] >= 0.9)
        merch_s, cnp_s = w.picker.pick(rng, mcc_s, city[ident], cnp_s)
        cnp_s = cnp_s | (M["mode"][merch_s] == "cnp")
        mu, sg = self._mu_sigma(merch_s)
        val_s = np.exp(mu + 0.6 + 0.9 * sg * rng.normal(0, 1, len(ident)))
        mcc_m = M["mcc_idx"][merch_s]
        phys = (mcc_m >= 0) & w.mccs["physical"].to_numpy()[mcc_m.clip(0)] & cnp_s
        kept = self.rows.add(len(ident), t=ts, customer=ids[ident], card=c_ids[ident], merchant=merch_s,
                             terminal=pick_terminal(rng, w, merch_s, ~cnp_s),
                             channel=np.where(cnp_s, "CNP", "CP").astype(object),
                             pos_entry_mode=np.where(cnp_s, "ecom", rng.choice(["chip", "contactless"], len(ident))).astype(object),
                             value_usd=val_s, device=np.where(cnp_s, dev[ident], -1), ip=np.where(cnp_s, ip[ident], -1),
                             email_domain=np.where(cnp_s, C["email"][ids[ident]], None).astype(object),
                             shipping_country=np.where(phys, gc, -1),
                             cred_type=np.where(cnp_s, "owner", "card_present").astype(object),
                             is_fraud=1, scenario="synthetic_identity", incident=iid)
        C["abandon_t"][ids] = float(ts.max())
        return kept

    def merchant_collusion(self, t0: float) -> int:
        """A new merchant ramps volume with stolen and complicit cards, issues refunds to the
        complicit cards (refund abuse), triggers a chargeback spike and disappears."""
        w, rng = self.w, self.rng
        C, M, TT, cards = w.customers, w.merchants, w.terminals, w.cards
        g, gang = self._gang()
        R = float(rng.uniform(14, 40))
        names = ["digital_goods", "games", "electronics", "gift_cards", "misc_retail"]
        mcc = self.mcc_by_name[names[int(rng.integers(0, len(names)))]]
        if len(self.sleepers) and rng.random() < 0.5:
            # a long-standing merchant turns bad (bust-out merchant): old open date, sudden volume
            mid = int(rng.choice(self.sleepers))
            if M["kind"][mid] != "normal":
                return 0
            M["kind"][mid] = "collusive"
            mcc = int(M["mcc_idx"][mid])
        else:
            country = gang["country"] if rng.random() < 0.15 else 0
            city = int(w.city_of_country(rng, np.array([country]))[0])
            mid = int(M.append(mcc_idx=mcc, mode="cnp", city=city, uses_3ds=False, popularity=1.0,
                               term_start=len(TT), n_term=1, open_t=t0, kind="collusive")[0])
            TT.append(merchant=mid, virtual=True)
        v0, v1 = rng.uniform(1, 3), rng.uniform(6, 20)
        day_rate = v0 * (v1 / v0) ** (np.arange(int(R)) / max(R - 1, 1))
        n_d = rng.poisson(day_rate)
        n = int(n_d.sum())
        if n == 0:
            return 0
        t = (t0 + np.repeat(np.arange(int(R)), n_d) * DAY + rng.uniform(0, DAY, n)).astype(np.int64)
        t.sort()
        comp = [c for c in (self._random_card(t0, 5) for _ in range(int(rng.integers(5, 15)))) if c is not None]
        complicit = rng.random(n) < (0.4 if comp else 0.0)
        card = np.empty(n, np.int64)
        stolen_n = int((~complicit).sum())
        stolen_cards = [c for c in (self._any_card(t0) for _ in range(stolen_n)) if c is not None]
        if len(stolen_cards) < stolen_n:   # not enough eligible cards: every row becomes complicit/stolen from what exists
            pool_all = np.array(stolen_cards + comp, dtype=np.int64)
            if not len(pool_all):
                return 0
            card[:] = rng.choice(pool_all, n)
            complicit = np.isin(card, comp)
        else:
            card[~complicit] = np.array(stolen_cards, dtype=np.int64)
            if complicit.any():
                card[complicit] = rng.choice(np.array(comp, dtype=np.int64), int(complicit.sum()))
        owner = cards["owner"][card]
        mu, sg = self._mu_sigma(np.full(n, mid))
        value = np.exp(mu + 0.9 + sg * rng.normal(0, 1, n))
        value = np.where(rng.random(n) < 0.5, np.round(value) - 0.01, value)
        value = np.maximum(value, 1.0)
        dev = np.where(complicit, -1, rng.choice(gang["devices"], n))
        arm = self.tactics.choose(g, "ip")
        ip, ip_arm = self._ips_for_arm(gang, arm, n)
        cu = np.nonzero(complicit)[0]
        if len(cu):  # complicit cardholders buy from their own device and IP
            dev[cu] = current_device(rng, w, owner[cu].clip(0), t[cu])
            ip[cu] = home_ip(w, owner[cu].clip(0), t[cu])
        iid = self._new_incident("merchant_collusion", gang=g, start=int(t[0]), end=int(t[-1]), merchant=mid)
        kept = self.rows.add(n, t=t, customer=owner, card=card, merchant=mid, terminal=M["term_start"][mid],
                             channel="CNP", pos_entry_mode="ecom", value_usd=value, device=dev, ip=ip,
                             email_domain=np.where(complicit, C["email"][owner.clip(0)],
                                                   rng.choice(geo.FRAUD_EMAIL[0], n, p=geo.FRAUD_EMAIL[1])).astype(object),
                             cred_type=np.where(complicit, "owner", "stolen_full").astype(object),
                             is_fraud=1, scenario="merchant_collusion", incident=iid, gang=g, tac_ip=ip_arm)
        # refund abuse: money paid back to the complicit cards weeks later
        ref = np.nonzero(complicit & (rng.random(n) < 0.6))[0]
        if len(ref):
            tr = (t[ref] + rng.uniform(2, 20, len(ref)) * DAY).astype(np.int64)
            self.rows.add(len(ref), t=tr, customer=owner[ref], card=card[ref], merchant=mid,
                          terminal=M["term_start"][mid], channel="CNP", pos_entry_mode="ecom",
                          value_usd=value[ref] * rng.uniform(0.85, 1.0, len(ref)), device=-1, ip=-1,
                          cred_type="owner", is_fraud=1, scenario="merchant_collusion", incident=iid,
                          txn_type="refund", gang=g)
        return kept

    def transaction_laundering(self, t0: float) -> int:
        """A normal-looking front merchant processes stolen cards on behalf of an illicit one."""
        w, rng = self.w, self.rng
        C, M = w.customers, w.merchants
        if not len(self.online_low):
            return 0
        g, gang = self._gang()
        dom = self.online_low[self.m_country[self.online_low] == 0]
        m = int(rng.choice(dom if (len(dom) and rng.random() < 0.85) else self.online_low))   # cards are mostly domestic: so is the front
        dur = rng.uniform(20, 60) * DAY
        n = int(rng.integers(30, 90))
        t = (t0 + np.sort(rng.random(n)) * dur).astype(np.int64)
        t = t - (t % DAY) + np.where(rng.random(n) < 0.55, rng.choice([0, 1, 2, 3, 4, 5, 22, 23], n), rng.integers(0, 24, n)) * HOUR + rng.integers(0, HOUR, n)
        cards_ = [c for c in (self._any_card(t0) for _ in range(n)) if c is not None]
        if not cards_:
            return 0
        card = rng.choice(np.array(cards_, dtype=np.int64), len(t))
        price = rng.choice([29.99, 49.99, 59.0, 99.0, 149.0, 199.99], len(t))
        price = np.where(rng.random(len(t)) < 0.5, price, np.round(price * rng.lognormal(0, 0.6, len(t)), 2))
        arm = self.tactics.choose(g, "ip")
        ip, ip_arm = self._ips_for_arm(gang, arm, len(t))
        blend = rng.random(len(t)) < 0.5                  # half of the traffic is routed to look like ordinary shoppers
        ip_b, _ = self._ips_for_arm(gang, 1, len(t), clean=True)
        ip = np.where(blend, ip_b, ip)
        dev = np.where(blend, w.new_devices(rng, len(t), -1, "gang", False), rng.choice(gang["devices"], len(t)))
        emails = np.where(blend, [self._legit_email() for _ in range(len(t))], rng.choice(gang["emails"], len(t))).astype(object)
        M["kind"][m] = "front"
        iid = self._new_incident("transaction_laundering", gang=g, start=int(t[0]), end=int(t[-1]), merchant=m)
        return self.rows.add(len(t), t=t, customer=w.cards["owner"][card], card=card, merchant=m,
                             terminal=M["term_start"][m], channel="CNP", pos_entry_mode="ecom", value_usd=price,
                             device=dev, ip=ip,
                             email_domain=emails,
                             cred_type="stolen_full", is_fraud=1, scenario="transaction_laundering", incident=iid,
                             gang=g, tac_ip=ip_arm)

    def bust_out(self, t0: float) -> int:
        w, rng, cfg = self.w, self.rng, self.cfg
        C, M = w.customers, w.merchants
        ts, T = cfg.test_start_sec, self.T
        rate_med = float(np.median(C["rate"][: self.n_cust0]))
        cust = -1
        for _ in range(100):
            c = int(rng.integers(0, self.n_cust0))
            if (C["open_t"][c] < ts - 120 * DAY and C["rate"][c] > rate_med and not C["is_mule"][c]
                    and not (C["abandon_t"][c] < T)):
                cust = c
                break
        if cust < 0:
            return 0
        dur = rng.uniform(1, 4) * DAY
        n = int(rng.integers(8, 31))
        t = (t0 + np.sort(rng.random(n)) * dur).astype(np.int64)
        t = t[t < T]
        n = len(t)
        if n == 0:
            return 0
        names = ["electronics", "jewelry", "gift_cards", "department_store", "airlines", "crypto_quasi_cash",
                 "clothing", "misc_retail", "hotels"]
        mcc = rng.choice([self.mcc_by_name[x] for x in names], n, p=[0.25, 0.1, 0.1, 0.12, 0.08, 0.05, 0.12, 0.1, 0.08])
        cnp = (rng.random(n) < 0.5) | (w.mccs["cnp_share"].to_numpy()[mcc] >= 0.9)
        merchant, cnp = w.picker.pick(rng, mcc, np.full(n, C["city"][cust]), cnp)
        cnp = cnp | (M["mode"][merchant] == "cnp")
        terminal = pick_terminal(rng, w, merchant, ~cnp)
        mu, sg = self._mu_sigma(merchant)
        value = np.exp(mu + 0.7 + C["spend_offset"][cust] + 0.8 * sg * rng.normal(0, 1, n))
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
                             is_fraud=1, scenario="bust_out", incident=iid)
        C["abandon_t"][cust] = float(t[-1])
        return kept

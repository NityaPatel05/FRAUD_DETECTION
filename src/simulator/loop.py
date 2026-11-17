"""
Closed-loop engine: the world is simulated in DAILY EPOCHS and a pluggable ``DecisionPolicy``
decides every transaction (approve / step_up_3ds / review / decline) before its outcome exists.

Feedback implemented here
-------------------------
* declined rows lose their outcome (label gate A, see labels.py) and, for fraud, the attacker
  abandons the (incident, card) pair with some probability;
* step-up (3-D Secure) stops most fraud on stolen credentials (cardholder-only secrets), shifts
  liability to the issuer and costs legit customers a small abandonment rate;
* reviewed rows are decided by analysts (finite capacity, imperfect accuracy) and labelled by them;
* a reported fraud blocks the card at the reporting time and REISSUES a new card id (``prev_card_id``
  links them): attacker rows on the old card stop, legit rows move to the new card after a lag
  (merchant-on-file credentials go stale unless the account updater refreshes them: code 54);
* gangs observe approvals per tactic arm (ip kind / amount band / merchant group / card age) and
  shift effort towards what works (``scenarios.Tactics``);
* matured labels are delivered to the policy through ``observe`` (chargebacks arrive 1-120 days
  late; friendly fraud and unreported fraud are part of that stream), so a policy that retrains
  does so on exactly the labels a real system would have.

Policy interface (what Phase C plugs a trained model into)
----------------------------------------------------------
    class MyPolicy(DecisionPolicy):
        def observe(self, labels: pd.DataFrame, day: int) -> None: ...   # optional
        def decide(self, batch: EpochBatch) -> Decisions: ...

``batch.frame`` is the day's public transaction frame (pre-decision columns of assemble.PRE_COLUMNS,
plus ``hist_*`` causal history features), ``batch.history.frame()`` the previous days incl. decisions
and approvals (never labels), ``labels`` matured labels arrive only through ``observe``.
``Decisions(action, score, propensity=None)``: action in ACTIONS, score in [0, 1], optional
propensity (probability that this action was taken given the features; the engine multiplies in the
exploration propensity when ``cfg.explore_rate`` lets a would-be decline through).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import labels as lab
from .assemble import (DECISION_COLUMNS, PRE_COLUMNS, IdBook, IpStr, public_frame)
from .auth import apply_auth
from .config import SimConfig
from .entities import World
from .rows import Rows
from .scenarios import FraudFactory

DAY = 86_400
ACTIONS = ("approve", "step_up_3ds", "review", "decline")
BIG = np.int64(10) ** 10

# probability that a 3-D Secure challenge is passed, by what the payer actually holds
STEPUP_PASS = {"owner": None, "card_on_file": None, "card_present": None, "account_login": 0.35,
               "stolen_full": 0.22, "stolen_partial": 0.12, "enumerated_valid": 0.05, "invalid_pan": 0.0,
               "counterfeit": 0.0}
NO_BLOCK = {"mule_ring", "friendly_fraud"}


# ───────────────────────────── policy interface ─────────────────────────────
@dataclass
class Decisions:
    action: np.ndarray                       # object array of ACTIONS, one per batch row
    score: np.ndarray                        # risk score in [0, 1]
    propensity: np.ndarray | None = None     # P(action | features) of a stochastic policy


class HistoryBuffer:
    """The last ``days`` of decided rows (public columns + decision outcome, never labels)."""

    def __init__(self, days: int):
        self.days = days
        self.frames: list[pd.DataFrame] = []
        self.arr: list[dict] = []

    def add(self, frame: pd.DataFrame, arrays: dict) -> None:
        self.frames.append(frame)
        self.arr.append(arrays)
        if len(self.frames) > self.days:
            self.frames.pop(0)
            self.arr.pop(0)

    def frame(self) -> pd.DataFrame:
        return pd.concat(self.frames, ignore_index=True) if self.frames else pd.DataFrame(columns=PRE_COLUMNS)

    def arrays(self) -> dict | None:
        if not self.arr:
            return None
        return {k: np.concatenate([a[k] for a in self.arr]) for k in self.arr[0]}


@dataclass
class EpochBatch:
    day: int
    frame: pd.DataFrame                      # rows to decide (pre-decision columns + hist_* features)
    history: HistoryBuffer
    cfg: SimConfig = None


class DecisionPolicy:
    name = "policy"

    def observe(self, labels: pd.DataFrame, day: int) -> None:  # noqa: D401
        """Labels that matured at the start of ``day``: transaction_id, label_date, reported_label,
        label_status, dispute_reason. Default: ignore."""

    def decide(self, batch: EpochBatch) -> Decisions:
        raise NotImplementedError


class CallablePolicy(DecisionPolicy):
    """Wrap ``fn(frame) -> Decisions`` (or ``-> (action, score)``)."""

    def __init__(self, fn: Callable, observe_fn: Callable | None = None, name: str = "callable"):
        self.fn, self.observe_fn, self.name = fn, observe_fn, name

    def decide(self, batch: EpochBatch) -> Decisions:
        out = self.fn(batch.frame)
        return out if isinstance(out, Decisions) else Decisions(np.asarray(out[0], dtype=object), np.asarray(out[1], float))

    def observe(self, labels, day):
        if self.observe_fn is not None:
            self.observe_fn(labels, day)


class ScorePolicy(DecisionPolicy):
    """Thresholds over a model score: ``score_fn(frame) -> p``; decline >= t_decline, review >= t_review,
    step up >= t_stepup (card-not-present rows only)."""

    def __init__(self, score_fn: Callable, t_decline=0.9, t_review=0.6, t_stepup=0.2, observe_fn=None, name="score"):
        self.score_fn, self.observe_fn, self.name = score_fn, observe_fn, name
        self.t = (t_decline, t_review, t_stepup)

    def decide(self, batch: EpochBatch) -> Decisions:
        f = batch.frame
        p = np.asarray(self.score_fn(f), dtype=float)
        d, r, s = self.t
        action = np.where(p >= d, "decline", np.where(p >= r, "review",
                          np.where((p >= s) & (f["channel"].to_numpy() == "CNP"), "step_up_3ds", "approve")))
        return Decisions(action.astype(object), p)

    def observe(self, labels, day):
        if self.observe_fn is not None:
            self.observe_fn(labels, day)


class AllApprovePolicy(DecisionPolicy):
    """No fraud system at all (a baseline: what attackers face without any defence)."""
    name = "approve_all"

    def decide(self, batch: EpochBatch) -> Decisions:
        n = len(batch.frame)
        return Decisions(np.full(n, "approve", dtype=object), np.zeros(n))


class RulesPolicy(DecisionPolicy):
    """Default hand-written rules engine over current-row fields and causal history features."""
    name = "rules"

    def __init__(self, decline_at=0.20, review_at=0.07, stepup_at=0.012, strength: float = 1.0):
        self.decline_at, self.review_at, self.stepup_at, self.strength = decline_at, review_at, stepup_at, strength

    def score(self, f: pd.DataFrame) -> np.ndarray:
        def g(c):
            col = f[c]
            return col.astype(float).to_numpy() if str(col.dtype) in ("Int64", "Int8") else col.to_numpy()

        cnp = g("channel") == "CNP"
        amt = g("amount_usd")
        z = np.full(len(f), -6.0)
        ipt = g("ip_type")
        z += 1.4 * (ipt == "hosting")
        z += 1.0 * (pd.notna(g("ip_country")) & (g("ip_country") != g("card_country")))
        avs, cvc, tds = g("avs_result"), g("cvc_check"), g("three_ds_result")
        z += 0.9 * (avs == "N") + 0.2 * ((avs == "U") & cnp)
        z += 1.8 * (cvc == "N") + 0.35 * ((cvc == "P") & (g("pos_entry_mode") == "ecom"))
        z += 2.6 * np.isin(tds, ["N", "R"])
        z += 0.7 * (amt > 250) + 0.9 * (amt > 800)
        z += 1.2 * (g("hist_card_n_1h") >= 3) + 0.9 * (g("hist_card_n_1h") >= 6)
        z += 1.0 * (g("hist_ip_n_1h") >= 4) + 0.7 * (g("hist_ip_n_24h") >= 12)
        z += 0.8 * (g("hist_device_n_24h") >= 8) + 0.6 * (g("hist_merchant_n_1h") >= 15)
        z += 1.0 * (g("hist_card_decl_7d") >= 2)
        la = np.log(np.maximum(amt, 0.01))
        zs = (la - g("hist_card_logamt_mean")) / 0.9
        z += 0.9 * (np.nan_to_num(zs, nan=0.0) > 2.5)
        z += 0.9 * ((g("hist_card_n_14d") >= 3) & (g("hist_card_dev_n_14d") == 0) & cnp & f["device_id"].notna().to_numpy())
        cust = f["customer_id"].isna().to_numpy()
        z += 0.4 * (cust & cnp)
        age = g("account_age_days")
        z += 0.8 * ((np.nan_to_num(age, nan=999) < 7) & (amt > 150))
        z += 0.6 * (pd.notna(g("shipping_country")) & (g("shipping_country") != g("card_country")))
        z += 0.5 * ((np.nan_to_num(g("merchant_age_days"), nan=999) < 30) & cnp & (amt > 100))
        z -= 3.0 * (g("recurring") == 1)
        z = z * self.strength - (1 - self.strength) * 0.0
        return 1.0 / (1.0 + np.exp(-z))

    def decide(self, batch: EpochBatch) -> Decisions:
        f = batch.frame
        p = self.score(f)
        cnp = f["channel"].to_numpy() == "CNP"
        can_step = cnp & (f["three_ds_result"].to_numpy() == "not_used")
        action = np.where(p >= self.decline_at, "decline", np.where(p >= self.review_at, "review",
                          np.where((p >= self.stepup_at) & can_step, "step_up_3ds", "approve")))
        return Decisions(action.astype(object), p)


# ───────────────────────────── causal history features ─────────────────────────────
def _codes(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    out = np.full(len(x), -1, dtype=np.int64)
    m = x >= 0
    if m.any():
        out[m] = pd.factorize(x[m])[0]
    return out


def _prior_counts(key: np.ndarray, t: np.ndarray, windows: list[int]) -> dict:
    """For each row: number of EARLIER rows (in array order for equal t) with the same key within w seconds."""
    valid = key >= 0
    comp = np.where(valid, key * BIG + t, -1 - np.arange(len(key)))
    order = np.argsort(comp, kind="stable")
    cs = comp[order]
    pos = np.empty(len(key), dtype=np.int64)
    pos[order] = np.arange(len(key))
    return {w: np.where(valid, pos - np.searchsorted(cs, comp - w, side="left"), 0) for w in windows}


def hist_features(cur: dict, prev: dict | None) -> dict:
    """Causal aggregates for the rows of an epoch from the previous epochs + earlier rows of the same epoch."""
    n = len(cur["t"])
    pn = 0 if prev is None else len(prev["t"])
    cat = (lambda k: cur[k]) if prev is None else (lambda k: np.concatenate([prev[k], cur[k]]))
    t = cat("t").astype(np.int64)
    out = {}
    keys = {k: _codes(cat(k)) for k in ("card", "ip", "device", "merchant", "customer")}
    spec = {"card": [3600, DAY, 7 * DAY, 14 * DAY], "ip": [3600, DAY], "device": [DAY],
            "merchant": [3600, DAY, 7 * DAY], "customer": [DAY]}
    names = {3600: "1h", DAY: "24h", 7 * DAY: "7d", 14 * DAY: "14d"}
    for k, wins in spec.items():
        res = _prior_counts(keys[k], t, wins)
        for w in wins:
            out[f"hist_{k}_n_{names[w]}"] = res[w][pn:]
    pair = _codes(np.where((cat("card") >= 0) & (cat("device") >= 0), cat("card") * 10_000_000 + cat("device"), -1))
    out["hist_card_dev_n_14d"] = _prior_counts(pair, t, [14 * DAY])[14 * DAY][pn:]
    # declines in the previous days and mean log amount of the card (prev rows only: outcomes of
    # today's earlier rows are not known yet)
    decl = np.full(n, 0)
    mean = np.full(n, np.nan)
    if prev is not None and pn:
        ck = keys["card"]
        pk, ckn = ck[:pn], ck[pn:]
        tp, tc = t[:pn], t[pn:]
        d_mask = (pk >= 0) & (prev["approved"] == 0)
        comp_p = np.sort(pk[d_mask] * BIG + tp[d_mask])
        c_comp = ckn * BIG + tc
        dc = np.searchsorted(comp_p, c_comp, side="left") - np.searchsorted(comp_p, c_comp - 7 * DAY, side="left")
        decl = np.where(ckn >= 0, dc, 0)
        a_mask = (pk >= 0) & (prev["approved"] == 1)
        if a_mask.any():
            size = int(ck.max()) + 1
            s = np.bincount(pk[a_mask], weights=prev["logamt"][a_mask], minlength=size)
            c = np.bincount(pk[a_mask], minlength=size)
            ok = (ckn >= 0) & (c[ckn.clip(0)] >= 2)
            mean = np.where(ok, s[ckn.clip(0)] / np.maximum(c[ckn.clip(0)], 1), np.nan)
    out["hist_card_decl_7d"] = decl
    out["hist_card_logamt_mean"] = mean
    return out


# ───────────────────────────── the engine ─────────────────────────────
class ClosedLoop:
    def __init__(self, world: World, legit: dict, ff: FraudFactory, policy: DecisionPolicy,
                 rng: np.random.Generator, idb: IdBook, ipstr: IpStr):
        self.w, self.cfg, self.legit, self.ff, self.policy, self.rng = world, world.cfg, legit, ff, policy, rng
        self.idb, self.ipstr = idb, ipstr
        self.start = pd.Timestamp(self.cfg.start_date, tz="UTC")
        self.n_days = self.cfg.n_days
        self.lg_idx = np.searchsorted(legit["t"], np.arange(self.n_days + 1) * DAY)
        self.history = HistoryBuffer(self.cfg.history_days)
        self.refund_q: dict[int, list] = {}
        self.label_q: dict[int, list] = {}
        self.next_uid = 0
        self.n_elig = 0.0
        self.n_disp = 0.0
        self.out: list[pd.DataFrame] = []
        self.stats = dict(lost_legit_rows=0, stale_card_declines=0, review_overflow=0, cards_blocked=0,
                          cards_reissued=0, rows_total=0, friendly_disputes=0, refunds=0)
        self.daily: list[dict] = []
        self.pos: list[dict] = []          # internal positions per row (used by the events builder)

    # ── row assembly ──
    @staticmethod
    def _frame(d: dict | None) -> pd.DataFrame | None:
        if d is None or not len(d["t"]):
            return None
        r = Rows()
        r.add_frame(d)
        return r.to_frame()

    def _gate_rows(self, core: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Drop rows of abandoned accounts / blocked cards, re-point legit rows to reissued cards.
        Returns (rows, stale_mask)."""
        w, cfg, rng = self.w, self.cfg, self.rng
        C, K = w.customers, w.cards
        n = len(core)
        t = core["t"].to_numpy()
        cust = core["customer"].to_numpy()
        scen = core["scenario"].to_numpy()
        legit_like = np.isin(scen, ["legit", "account_takeover"])
        drop = np.zeros(n, bool)
        ab = C["abandon_t"][cust.clip(0)]
        drop |= (scen == "legit") & (cust >= 0) & (t > ab)
        card = core["card"].to_numpy().copy()
        rec = core["recurring"].to_numpy() == 1
        stale = np.zeros(n, bool)
        active = ~drop
        lost = 0
        for _ in range(6):
            bt = K["blocked_t"][card]
            blk = active & (bt < t)
            if not blk.any():
                break
            rt, succ = K["reissue_t"][card], K["succ"][card]
            gone = blk & ~legit_like
            wait = blk & legit_like & ((succ < 0) | ~(t >= rt))
            lost += int((wait & (scen == "legit")).sum())
            mv = blk & legit_like & (succ >= 0) & (t >= rt)
            st = mv & rec & (rng.random(n) > cfg.account_updater_share)
            stale |= st
            mv &= ~st
            card[mv] = succ[mv]
            active &= ~(gone | wait | st)
        self.stats["lost_legit_rows"] += lost
        core = core.copy()
        core["card"] = card
        keep = ~drop & (active | stale)
        return core[keep].reset_index(drop=True), stale[keep]

    def _hist_arrays(self, core: pd.DataFrame) -> dict:
        return dict(t=core["t"].to_numpy(), card=core["card"].to_numpy(), ip=core["ip"].to_numpy(),
                    device=core["device"].to_numpy(), merchant=core["merchant"].to_numpy(),
                    customer=np.where(core["guest"].to_numpy() == 1, -1, core["customer"].to_numpy()))

    # ── one epoch ──
    def run_epoch(self, d: int) -> None:
        w, cfg, rng, ff = self.w, self.cfg, self.rng, self.ff
        C, K = w.customers, w.cards
        t0, t1 = d * DAY, (d + 1) * DAY
        lbl = self.label_q.pop(d, None)
        if lbl:
            self.policy.observe(pd.concat(lbl, ignore_index=True), d)
        ff.step(d)
        parts = [self._frame({k: v[self.lg_idx[d]:self.lg_idx[d + 1]] for k, v in self.legit.items()}),
                 self._frame(ff.take(t1)), self._frame(self._pop_refunds(d))]
        parts = [p for p in parts if p is not None]
        if not parts:
            ff.end_of_day()
            return
        core = pd.concat(parts, ignore_index=True)
        core, stale = self._gate_rows(core)
        if not len(core):
            ff.end_of_day()
            return
        order = np.argsort(core["t"].to_numpy(), kind="stable")
        core, stale = core.iloc[order].reset_index(drop=True), stale[order]
        n = len(core)
        core["uid"] = self.next_uid + np.arange(n, dtype=np.int64)
        self.next_uid += n

        avs, cvc, tds, code = apply_auth(core, w, rng)
        refund = core["txn_type"].to_numpy() == "refund"
        code = np.where(stale, "54", code).astype(object)
        self.stats["stale_card_declines"] += int(stale.sum())
        code[refund] = "00"
        avs[refund] = None
        cvc[refund] = None
        tds[refund] = None
        pub = public_frame(core, w, cfg, self.idb, self.ipstr, avs, cvc, tds)

        # ── decisions ──
        dmask = ~refund
        di = np.nonzero(dmask)[0]
        arr_all = self._hist_arrays(core)
        hist = hist_features({k: v[di] for k, v in arr_all.items()}, self.history.arrays())
        frame = pub.iloc[di].reset_index(drop=True)
        for k, v in hist.items():
            frame[k] = v
        dec = self.policy.decide(EpochBatch(d, frame, self.history, cfg))
        action = np.asarray(dec.action, dtype=object)
        if len(action) != len(di) or not np.isin(action, ACTIONS).all():
            raise ValueError("policy returned a wrong number of actions or an unknown action")
        outcome = self._apply(core, pub, code, tds, di, action, np.asarray(dec.score, float), dec.propensity, stale)
        self._label_and_finish(d, core, pub, outcome, arr_all, di)

    def _pop_refunds(self, d: int):
        q = self.refund_q.pop(d, None)
        if not q:
            return None
        return {k: np.concatenate([x[k] for x in q]) for k in q[0]}

    def _apply(self, core, pub, issuer_code, tds, di, action, score, prop, stale) -> dict:
        cfg, rng = self.cfg, self.rng
        n, m = len(core), len(di)
        act = np.full(n, "approve", dtype=object)
        act[di] = action
        sc = np.full(n, np.nan)
        sc[di] = score
        pr = np.ones(n)
        if prop is not None:
            pr[di] = np.asarray(prop, float)
        eps = cfg.explore_rate
        wd = act == "decline"
        ex = np.zeros(n, bool)
        if eps > 0:
            ex = wd & (rng.random(n) < eps)
            pr = np.where(wd, np.where(ex, eps, 1 - eps), pr)
        eff = np.where(ex, "approve", act)
        channel = core["channel"].to_numpy()
        cred = core["cred_type"].to_numpy()
        cnp = channel == "CNP"
        # 3-D Secure step-up
        su = (eff == "step_up_3ds") & cnp & (tds == "not_used")
        pp = np.array([STEPUP_PASS.get(c, None) if STEPUP_PASS.get(c, None) is not None else cfg.stepup_legit_pass
                       for c in cred]) if su.any() else np.zeros(n)
        su_pass = su & (rng.random(n) < pp)
        su_fail = su & ~su_pass
        tds_f = tds.copy()
        tds_f[su_pass] = "C"
        tds_f[su_fail] = "N"
        # manual review: finite analyst capacity, imperfect accuracy
        rv = eff == "review"
        reviewed = np.zeros(n, bool)
        if rv.any():
            cap = max(1, int(np.ceil(cfg.review_capacity_frac * m)))
            idx = np.nonzero(rv)[0]
            top = idx[np.argsort(-sc[idx])[:cap]]
            reviewed[top] = True
            self.stats["review_overflow"] += int(len(idx) - len(top))
        fraud = core["is_fraud"].to_numpy() == 1
        caught = reviewed & fraud & (rng.random(n) < cfg.review_accuracy)
        wrong = reviewed & ~fraud & (rng.random(n) < 0.03)
        review_decl = caught | wrong
        issuer_ok = issuer_code == "00"
        declined_by = np.where(wd & ~ex, "policy", np.where(su_fail, "sca", np.where(review_decl, "review",
                               np.where(~issuer_ok, "issuer", "none")))).astype(object)
        refund = core["txn_type"].to_numpy() == "refund"
        declined_by[refund] = "none"
        approved = declined_by == "none"
        code = np.where(declined_by == "policy", "59", np.where(declined_by == "sca", "1A",
                        np.where(declined_by == "review", "59", issuer_code))).astype(object)
        code[approved] = "00"
        liab = (approved & cnp & np.isin(tds_f, ["Y", "C"])).astype(np.int8)
        return dict(action=eff, score=sc, prop=pr, declined_by=declined_by, approved=approved, code=code,
                    tds=tds_f, liab=liab, reviewed=reviewed, review_fraud=caught | wrong)

    def _label_and_finish(self, d, core, pub, o, arr_all, di):
        cfg, rng, w, ff = self.cfg, self.rng, self.w, self.ff
        C, K = w.customers, w.cards
        n = len(core)
        t = core["t"].to_numpy()
        cust = core["customer"].to_numpy()
        card = core["card"].to_numpy()
        scen = core["scenario"].to_numpy().astype(object)
        role = core["role"].to_numpy()
        fraud = core["is_fraud"].to_numpy() == 1
        approved = o["approved"]
        kind = C["kind"][cust.clip(0)]
        real_owner = (cust >= 0) & (kind != "phantom")
        res = lab.assign(rng, cfg, t=t, is_fraud=core["is_fraud"].to_numpy(), scenario=scen, role=role,
                         approved=approved, declined_by=o["declined_by"], decision=o["action"],
                         real_owner=real_owner, channel=core["channel"].to_numpy(),
                         txn_type=core["txn_type"].to_numpy(), reviewed_fraud=o["review_fraud"], reviewed=o["reviewed"])
        status, reported, label_t, report_t, cb_t, dispute = (res[k] for k in
                                                             ("status", "reported", "label_t", "report_t", "cb_t", "dispute"))
        # friendly fraud: legit approved e-commerce purchases disputed as fraud
        value = core["value_usd"].to_numpy()
        elig = (approved & (scen == "legit") & (core["channel"].to_numpy() == "CNP") & ~fraud
                & (core["txn_type"].to_numpy() == "purchase") & (core["recurring"].to_numpy() == 0) & (value > 10))
        p_f = lab.friendly_probability(cfg, self.n_disp, self.n_elig)
        fr = elig & (rng.random(n) < p_f)
        self.n_elig += float(elig.sum())
        if fr.any():
            cbd = np.clip(rng.lognormal(np.log(cfg.cb_delay_median_days), cfg.cb_delay_sigma, n), 2, cfg.label_window_days) * DAY
            rev = fr & (rng.random(n) < cfg.friendly_reversal_rate)
            status[fr] = "chargeback"
            reported[fr] = 1.0
            label_t[fr] = (t + cbd)[fr]
            cb_t[fr] = (t + cbd)[fr]
            dispute[fr] = np.where(rng.random(n) < 0.75, "fraud_cnp", "fraud_unauthorized")[fr]
            status[rev] = "reversed"
            reported[rev] = 0.0
            label_t[rev] = (t + cbd + rng.uniform(10, 40, n) * DAY)[rev]
            scen = scen.copy()
            scen[fr] = "friendly_fraud"
            self.stats["friendly_disputes"] += int(fr.sum())
        self.n_disp += float((fraud & (reported == 1)).sum())

        # ── card blocks & reissues (the victim reports, or an analyst / cardholder contact confirms) ──
        blk_ok = np.isfinite(report_t) & ~np.isin(scen, list(NO_BLOCK)) & (core["txn_type"].to_numpy() != "refund") & fraud
        if blk_ok.any():
            self._block_cards(card[blk_ok], report_t[blk_ok])

        # ── attacker feedback ──
        gang = core["gang"].to_numpy()
        gm = (gang >= 0) & fraud & (core["txn_type"].to_numpy() == "purchase")
        if gm.any():
            ff.feedback(gang[gm], core["tac_ip"].to_numpy()[gm], core["tac_amt"].to_numpy()[gm],
                        core["tac_mg"].to_numpy()[gm], core["tac_aged"].to_numpy()[gm], approved[gm])
            gone = gm & ~approved & (rng.random(n) < 0.65)
            if gone.any():
                ff.abandon(core["incident"].to_numpy()[gone], card[gone])
        ff.end_of_day()

        # ── schedule refunds of approved legit purchases ──
        rp = np.where(core["channel"].to_numpy() == "CNP", cfg.refund_rate, cfg.refund_rate * 0.25)
        cand = (approved & (scen == "legit") & (core["txn_type"].to_numpy() == "purchase")
                & (core["recurring"].to_numpy() == 0) & (core["channel"].to_numpy() != "P2P") & (rng.random(n) < rp))
        if cand.any():
            ci = np.nonzero(cand)[0]
            delay = np.clip(rng.lognormal(np.log(6.0), 0.9, len(ci)), 0.5, 40.0) * DAY
            tr = (t[ci] + delay).astype(np.int64)
            full = rng.random(len(ci)) < 0.7
            day_r = tr // DAY
            ok = day_r < self.n_days
            cols = dict(t=tr, customer=cust[ci], card=card[ci], merchant=core["merchant"].to_numpy()[ci],
                        terminal=core["terminal"].to_numpy()[ci], channel=core["channel"].to_numpy()[ci],
                        pos_entry_mode=core["pos_entry_mode"].to_numpy()[ci],
                        value_usd=np.where(full, value[ci], value[ci] * rng.uniform(0.3, 0.9, len(ci))),
                        cred_type=np.full(len(ci), "owner", dtype=object), txn_type=np.full(len(ci), "refund", dtype=object),
                        refund_of=core["uid"].to_numpy()[ci], guest=core["guest"].to_numpy()[ci])
            for dd in np.unique(day_r[ok]):
                sel = ok & (day_r == dd)
                self.refund_q.setdefault(int(dd), []).append({k: v[sel] for k, v in cols.items()})
            self.stats["refunds"] += int(ok.sum())

        # ── queue label deliveries ──
        fin = np.isfinite(label_t)
        if fin.any():
            ld = np.maximum((np.nan_to_num(label_t, nan=0.0) // DAY).astype(np.int64) + 1, d + 1)   # known from the next midnight
            ok = fin & (ld < self.n_days)
            if ok.any():
                start = self.start
                lab_df = pd.DataFrame({
                    "transaction_id": core["uid"].to_numpy() + 1,
                    "label_date": start + pd.to_timedelta(np.where(fin, label_t, 0), unit="s"),
                    "reported_label": pd.arrays.IntegerArray(np.nan_to_num(reported).astype(np.int8), np.isnan(reported)),
                    "label_status": status, "dispute_reason": dispute})
                for dd in np.unique(ld[ok]):
                    self.label_q.setdefault(int(dd), []).append(lab_df[ok & (ld == dd)])

        # ── final per-row frame ──
        start = self.start
        out = pub.copy()
        out["three_ds_result"] = o["tds"]
        out["decision"] = o["action"]
        out["risk_score"] = o["score"]
        out["propensity"] = o["prop"]
        out["declined_by"] = o["declined_by"]
        out["liability_shift"] = o["liab"]
        out["auth_response_code"] = o["code"]
        out["approved"] = approved.astype(np.int8)
        out["reported_label"] = pd.arrays.IntegerArray(np.nan_to_num(reported).astype(np.int8), np.isnan(reported))
        out["label_status"] = status
        out["label_date"] = start + pd.to_timedelta(label_t, unit="s")
        out["chargeback_date"] = start + pd.to_timedelta(np.where(np.isin(status, ["chargeback", "reversed"]), cb_t, np.nan), unit="s")
        out["dispute_reason"] = dispute
        tf = core["is_fraud"].to_numpy().astype(np.int8)
        out["is_fraud"] = tf
        out["is_fraud_true"] = tf
        out["fraud_scenario"] = scen
        out["incident_id"] = pd.arrays.IntegerArray(np.where(core["incident"].to_numpy() >= 0, core["incident"].to_numpy(), 0),
                                                    core["incident"].to_numpy() < 0)
        out["ring_id"] = pd.arrays.IntegerArray(np.where(core["ring"].to_numpy() >= 0, core["ring"].to_numpy(), 0),
                                                core["ring"].to_numpy() < 0)
        out["gang_id"] = pd.arrays.IntegerArray(np.where(gang >= 0, gang, 0), gang < 0)
        self.out.append(out)
        self.pos.append(dict(t=t, customer=cust, device=core["device"].to_numpy(), ip=core["ip"].to_numpy(),
                             counterparty=core["counterparty"].to_numpy(), guest=core["guest"].to_numpy()))
        self.stats["rows_total"] += n
        # history for the next epochs (policy rows only: refunds are not scored)
        hist_core = {k: v[di] for k, v in arr_all.items()}
        hist_core["approved"] = approved[di].astype(np.int8)
        hist_core["logamt"] = np.log(np.maximum(core["value_usd"].to_numpy()[di], 0.01))
        self.history.add(out.iloc[di][PRE_COLUMNS + DECISION_COLUMNS].reset_index(drop=True), hist_core)
        self.daily.append(dict(day=d, rows=n, fraud=int(fraud.sum()), approved_fraud=int((fraud & approved).sum()),
                               declined_legit=int((~fraud & ~approved).sum()), legit=int((~fraud).sum())))

    def _block_cards(self, card: np.ndarray, report_t: np.ndarray) -> None:
        cfg, rng, w = self.cfg, self.rng, self.w
        C, K = w.customers, w.cards
        g = pd.DataFrame({"card": card, "rt": report_t}).groupby("card")["rt"].min()
        cards, rts = g.index.to_numpy().astype(np.int64), g.to_numpy()
        cur = K["blocked_t"][cards]
        fresh = np.isnan(cur) | (rts < cur)
        cards, rts = cards[fresh], rts[fresh]
        if not len(cards):
            return
        K["blocked_t"][cards] = rts
        self.stats["cards_blocked"] += len(cards)
        owner = K["owner"][cards]
        can = (owner >= 0) & (C["kind"][owner.clip(0)] == "regular") & (K["succ"][cards] < 0)
        if can.any():
            rc = cards[can]
            rrt = rts[can] + rng.uniform(cfg.reissue_lag_days[0], cfg.reissue_lag_days[1], len(rc)) * DAY
            new = K.append(owner=K["owner"][rc], country=K["country"][rc], network=K["network"][rc],
                           card_type=K["card_type"][rc], bin=K["bin"][rc], prev=rc, issued_t=rrt)
            K["succ"][rc] = new
            K["reissue_t"][rc] = rrt
            self.stats["cards_reissued"] += len(rc)

    def run(self, verbose: bool = False) -> pd.DataFrame:
        self.ff.preroll()
        for d in range(self.n_days):
            self.run_epoch(d)
            if verbose and d % 60 == 0:
                print(f"[sim] epoch {d}/{self.n_days}  rows so far {self.stats['rows_total']:,}")
        return pd.concat(self.out, ignore_index=True)

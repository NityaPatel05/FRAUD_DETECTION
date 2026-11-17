"""
Phase 7 — online feature engine: the real-time twin of the offline v2 pipeline.

Production practice (Feedzai Railgun, Stripe Chronon/Shepherd, Uber Palette): one
feature definition, computed online from per-entity state, with a PARITY test that
replays history and checks the online values equal the offline training values.
Train/serve skew is one of the most common silent failures in fraud ML.

Design
  * Row-level features (cleaning, D-normalisation, derived signals, uid keys) reuse the
    exact offline functions on a 1-row (or batch) DataFrame → identical by construction.
  * Stateful features (velocity, aggregates, EMA, lags, HMM windows, sharing counts,
    delayed labels, identity clusters) are recomputed from compact per-entity state:
    READ the state as it was before this event, then UPDATE it.
  * Labels follow the shared policy in src/labels/policy.py (meta["label_policy"]):
      - every scored transaction registers a MATURITY event at t + maturity_days; at that time it
        counts as a labelled row (n) -- a mature transaction with no fraud report is a negative;
      - record_label(txn_id, 1) (a chargeback) registers FRAUD events at the time the policy says
        the label is known (t + fraud_delay(txn_id)); no explicit negatives are ever needed;
      - events are popped from one global heap when the clock (event time of the next scored
        transaction) reaches them, so features equal the offline vectorised ones exactly.
  * Concurrency: one re-entrant engine lock serialises every state read/update (feature
    computation is ~1-3 ms of pure Python, so throughput is bounded to ~1/latency on one core;
    scale out by sharding on uid, not by threads). Row-level preprocessing and model inference
    run outside the lock.
  * Idempotency: a TransactionID already scored never mutates state again (see featurize).
  * Bounded state: windows are dropped when idle, idle entity profiles after profile_ttl_days,
    the txn index after the longest label delay, the decision cache by size (see prune()).
  * Only agg_mode = causal is servable (transductive needs the future).

Usage
    eng = OnlineFeatureEngine(meta)                  # meta.json from run_feature_engineering
    eng.replay(history_raw_df, labels=True)          # warm state from history (time-sorted)
    x = eng.transform(txn_dict)                      # → pd.Series in meta["feature_cols"] order
    eng.record_label(txn_id, 1)                      # when a chargeback lands (fraud only)
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import pickle
import threading
from collections import OrderedDict, defaultdict, deque
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.feature_engineering.base_features import build_base_features
from src.feature_engineering.client_features import (
    build_uid_keys, C_FOR_MEAN, M_FOR_MEAN, UID_MEAN_STD, UID_NUNIQUE, COUNT_ENCODE, FIRST_SEEN_VALUES,
)
from src.feature_engineering.derived_signals import add_derived_signals
from src.feature_engineering.graph_features import SHARED_ENTITY_PAIRS
from src.feature_engineering.hmm_features import HMM_PERSPECTIVES, hmm_surprise_from_window
from src.feature_engineering.label_history import LABEL_KEYS
from src.feature_engineering.derived_signals import vendor_engineered_cols
from src.feature_engineering.pipeline_v2 import _downcast
from src.labels.policy import DAY, LabelPolicy, fraud_known_at, maturity_at

_FE_MODULES = ["src.feature_engineering.base_features", "src.feature_engineering.client_features",
               "src.feature_engineering.derived_signals"]


def _isnan(v) -> bool:
    if v is None or v is pd.NA:
        return True
    try:
        return bool(isinstance(v, float) and math.isnan(v)) or (isinstance(v, np.floating) and np.isnan(v))
    except TypeError:
        return False


def _deque3():
    return deque(maxlen=3)


def _deque2():
    return deque(maxlen=2)


def _key(v):
    """Hashable entity key; None when missing (missing keys never form an entity)."""
    if _isnan(v):
        return None
    if isinstance(v, (np.floating, float)):
        return float(np.float32(v)) if isinstance(v, np.float32) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _num(v) -> float:
    return float("nan") if _isnan(v) else float(v)


class _Window:
    """Exact sliding window (t - w, t] with count / sum / distinct support."""
    __slots__ = ("q", "w", "s", "cnt")

    def __init__(self, w):
        self.q, self.w, self.s, self.cnt = deque(), w, 0.0, defaultdict(int)

    def push(self, t, x=0.0, item=None):
        self.q.append((t, x, item)); self.s += x
        if item is not None:
            self.cnt[item] += 1

    def evict(self, t):
        while self.q and self.q[0][0] <= t - self.w:
            _, x, item = self.q.popleft(); self.s -= x
            if item is not None:
                self.cnt[item] -= 1
                if self.cnt[item] == 0:
                    del self.cnt[item]


class _Agg:
    __slots__ = ("n", "s1", "s2")

    def __init__(self):
        self.n = 0.0; self.s1 = 0.0; self.s2 = 0.0

    def add(self, x):
        if not math.isnan(x):
            self.n += 1; self.s1 += x; self.s2 += x * x

    def mean(self):
        return self.s1 / self.n if self.n > 0 else float("nan")

    def std(self):
        if self.n <= 1:
            return float("nan")
        var = (self.s2 - self.s1 * self.s1 / self.n) / (self.n - 1)
        return math.sqrt(max(var, 0.0))


class _Ctr:
    """Label counters of one entity: n = mature rows, f = known frauds, fm = known frauds that are mature."""
    __slots__ = ("n", "f", "fm")

    def __init__(self):
        self.n = 0.0; self.f = 0.0; self.fm = 0.0


_EV_MATURE, _EV_FRAUD, _EV_FRAUD_MATURE = 0, 1, 2


class DuplicateTransactionError(ValueError):
    """The TransactionID was already scored: state must not be mutated twice."""


class SnapshotMismatchError(RuntimeError):
    """Engine snapshot was written for a different model / meta / feature list / label policy."""


class MissingFieldsError(ValueError):
    """The bundle needs input fields the gateway did not send (e.g. Vesta-only columns)."""


class _DSU:
    def __init__(self):
        self.parent, self.size, self.fraud, self.lab = {}, {}, defaultdict(float), defaultdict(float)
        self.deg = defaultdict(float)
        self.fraud_m = defaultdict(float)

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1.0 if x[0] == "c" else 0.0
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[rb] > self.size[ra]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]; self.fraud[ra] += self.fraud[rb]; self.lab[ra] += self.lab[rb]
        self.fraud_m[ra] += self.fraud_m[rb]


def resolve_policy(meta: Dict) -> LabelPolicy:
    """The label policy the features were BUILT with (meta) -- never a serving-side guess."""
    if meta.get("label_policy"):
        return LabelPolicy.from_cfg(meta["label_policy"])
    if meta["fe_cfg"].get("label_policy"):
        return LabelPolicy.from_cfg(meta["fe_cfg"]["label_policy"])
    if meta["fe_cfg"].get("label_delay_days"):           # pre-policy meta: fixed delay for every label
        return LabelPolicy.fixed(float(meta["fe_cfg"]["label_delay_days"]))
    raise ValueError("meta has no label_policy: rebuild the features (run_feature_engineering)")


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class OnlineFeatureEngine:
    """serving: optional dict -- profile_ttl_days (400), prune_every (txns between sweeps, 20000),
    decision_cache_size (100000), label_grace_days (30), require_vendor_fields (True)."""

    def __init__(self, meta: Dict, serving: Optional[Dict] = None):
        if meta["agg_mode"] != "causal":
            raise ValueError("Only agg_mode=causal features can be served online")
        self.meta = meta
        self.cfg = meta["fe_cfg"]
        self.feature_cols: List[str] = meta["feature_cols"]
        self.policy = resolve_policy(meta)
        self.smooth = float(self.cfg["label_smoothing"])
        self.prior = float(meta["train_fraud_rate"])
        self.cat_index = {c: {v: i for i, v in enumerate(cats)} for c, cats in meta["label_maps"].items()}
        self.hmm = {k: tuple(np.asarray(a, dtype=np.float64) for a in v) for k, v in meta.get("hmm_params", {}).items()}
        sv = dict(serving or {})
        longest_window = max([int(w) for w in self.cfg["velocity_windows_sec"]] + [30 * DAY, int(self.cfg["count_window_days"]) * DAY])
        self.window_ttl = longest_window + DAY                        # idle longer than this => every window is empty
        ptl = sv.get("profile_ttl_days", 400)
        self.profile_ttl = None if ptl is None else max(int(float(ptl) * DAY), self.window_ttl)
        self.prune_every = int(sv.get("prune_every", 20000))
        self.decision_cache_size = int(sv.get("decision_cache_size", 100_000))
        self.index_ttl = int((max(self.policy.delay_cap_days, self.policy.maturity_days)
                              + float(sv.get("label_grace_days", 30))) * DAY)
        # Vesta-only columns a real gateway never sends (only relevant for schema=full bundles)
        self.vendor_fields = vendor_engineered_cols(meta["raw_columns"]) if meta.get("schema") == "full" else []
        self.require_vendor = bool(sv.get("require_vendor_fields", True))
        self._lock = threading.RLock()
        self._reset_state()

    # ── fingerprint (snapshot validation) ────────────────────────────────────
    def fingerprint(self) -> Dict:
        m = self.meta
        return {"feature_cols": _hash(self.feature_cols), "raw_columns": _hash(m["raw_columns"]),
                "label_policy": _hash(self.policy.to_dict()),
                "fe_cfg": _hash({k: v for k, v in self.cfg.items() if k != "drop_cols"}),
                "label_maps": _hash({k: list(v) for k, v in m["label_maps"].items()}),
                "hmm": _hash({k: [np.asarray(a).tolist() for a in v] for k, v in m.get("hmm_params", {}).items()}),
                "prior": round(float(m["train_fraud_rate"]), 12), "schema": m.get("schema"), "tag": m.get("tag")}

    # ── state ────────────────────────────────────────────────────────────────
    def _reset_state(self):
        dd = lambda f: defaultdict(f)                                   # noqa: E731
        self.win = dd(dict)            # (ns, key) -> {window_sec: _Window}
        self.agg = dd(_Agg)            # (ns, key, col) -> _Agg
        self.seen = dd(set)            # (ns, key) -> set(values)
        self.count = dd(float)         # (ns, key) -> number of rows so far
        self.first_t = {}              # (ns, key) -> first event time
        self.hist = dd(_deque3)        # uid -> last 3 rows (dict)
        self.ema = {}                  # (uid, hl) -> [c, s, s2, last]
        self.hmm_x = dd(_deque2)       # (persp, obs, key) -> last 2 x
        self.hmm_last_t = {}           # (persp, key) -> previous t
        self.delayed = dd(_Ctr)        # (ns, key) -> label counters (n, f, fm)
        self.dsu = _DSU()
        self.events: list = []         # heap (ready_t, seq, kind, keys, uid): maturity / fraud-known events
        self.browser_newest: Dict = {}
        self.txn_keys: Dict = {}       # txn_id -> (t, {ns: key}, uid): the TTL-bounded transaction index
        self.index_order: deque = deque()   # (t, txn_id) in arrival order, for TTL pruning
        self.fraud_ids: set = set()    # txn ids whose fraud label was registered (idempotent labels)
        self.decisions: OrderedDict = OrderedDict()   # txn_id -> decision (idempotent re-scoring)
        self.touched: OrderedDict = OrderedDict()      # entity key value -> last event time
        self.clock = 0                 # largest event time seen
        self.n_scored = 0
        self.label_stats: Dict[str, int] = defaultdict(int)
        self._seq = 0

    # ── row-level (shared with offline) ──────────────────────────────────────
    def row_frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        cols = self.meta["raw_columns"]
        rows = rows.rename(columns=lambda c: c.replace("id-", "id_") if c.startswith("id-") else c)
        if rows.columns.duplicated().any():          # train "id_19" + Kaggle-test "id-19" concatenated
            merged = {c: rows.loc[:, rows.columns == c].bfill(axis=1).iloc[:, 0]
                      for c in rows.columns[rows.columns.duplicated()].unique()}
            rows = rows.loc[:, ~rows.columns.duplicated()].copy()
            for c, v in merged.items():
                rows[c] = v
        df = rows.reindex(columns=cols).copy()
        for c, dt in self.meta["raw_dtypes"].items():
            try:
                df[c] = df[c].astype(dt)
            except (ValueError, TypeError):
                df[c] = pd.to_numeric(df[c], errors="coerce") if dt.startswith(("int", "float")) else df[c]
        for m in _FE_MODULES:
            logger.disable(m)
        try:
            build_base_features(df)
            build_uid_keys(df)
            df = _downcast(df)
            df = df.drop(columns=[c for c in self.meta.get("dropped_v_cols", []) if c in df.columns])
            if self.cfg.get("use_derived_signals", True):
                der = add_derived_signals(df)
                df = pd.concat([df, pd.DataFrame(der, index=df.index)], axis=1)
        finally:
            for m in _FE_MODULES:
                logger.enable(m)
        return df

    # ── helpers ──────────────────────────────────────────────────────────────
    def _window(self, ns, key, w):
        d = self.win[(ns, key)]
        if w not in d:
            d[w] = _Window(w)
        return d[w]

    def _first_time(self, ns, key, value):
        """offline first_time_pair: NaN if value missing or entity has no prior row."""
        if key is None or value is None:
            return float("nan")
        prior = self.count[(ns, key)] > 0
        new = value not in self.seen[(ns, key)]
        return float("nan") if not prior else float(new)

    # ── one event ────────────────────────────────────────────────────────────
    def _apply_events(self, now: int) -> None:
        """Count every label event that is due (ready_t <= now), exactly like the offline `ready <= t`."""
        ev = self.events
        use_dsu = self.cfg.get("use_network_v4", True)
        while ev and ev[0][0] <= now:
            _, _, kind, keys, uid = heapq.heappop(ev)
            for ns, k in keys.items():
                if k is not None:
                    c = self.delayed[(ns, k)]
                    if kind == _EV_MATURE:
                        c.n += 1
                    elif kind == _EV_FRAUD:
                        c.f += 1
                    else:
                        c.fm += 1
            if use_dsu and uid is not None:
                rt = self.dsu.find(("c", uid))
                if kind == _EV_MATURE:
                    self.dsu.lab[rt] += 1
                elif kind == _EV_FRAUD:
                    self.dsu.fraud[rt] += 1
                else:
                    self.dsu.fraud_m[rt] += 1

    def _stateful(self, r: Dict, labelled: bool = True) -> Dict[str, float]:
        """Features of one event from state BEFORE it, then update state. ``labelled=False`` (offline
        replay of unlabelled rows, e.g. Kaggle test) skips the maturity event: such rows never count."""
        out: Dict[str, float] = {}
        t = int(r["TransactionDT"]); amt = float(r["TransactionAmt"])
        self.clock = max(self.clock, t)
        self._apply_events(t)
        has = r.__contains__
        uid = _key(r.get("uid1"))
        cfg = self.cfg
        keys = {k: _key(r.get(k)) for k in set(COUNT_ENCODE + LABEL_KEYS + ["uid1", "uid2", "device_fp",
                "card1_addr1", "card1", "card1_addr1_P_emaildomain", "addr1", "P_emaildomain"]) if has(k)}

        # derived: browser version lag (causal running max per family)
        if "browser_version_lag" in self.feature_cols or "browser_version" in r:
            fam = r.get("browser_family"); ver = _num(r.get("browser_version"))
            famk = "NA" if _isnan(fam) else fam
            if not math.isnan(ver):
                self.browser_newest[famk] = max(self.browser_newest.get(famk, -math.inf), ver)
            newest = self.browser_newest.get(famk, float("nan"))
            out["browser_version_lag"] = (newest - ver) if not math.isnan(ver) and newest != -math.inf else float("nan")

        # count encoding: rows with this value in the trailing window (incl. current)
        wd = int(cfg["count_window_days"])
        for c in COUNT_ENCODE:
            if has(c):
                k = keys.get(c)
                if k is None:
                    out[f"{c}_count{wd}d"] = float("nan"); continue
                w = self._window("ce_" + c, k, wd * 86400); w.evict(t); w.push(t)
                out[f"{c}_count{wd}d"] = float(len(w.q))

        # client aggregates (expanding, incl. current)
        def agg(ns, key, col, x, stats=("mean", "std")):
            if key is None:
                for s_ in stats:
                    out[f"{ns}_{col}_{s_}"] = float("nan")
                return
            a = self.agg[(ns, key, col)]; a.add(x)
            for s_ in stats:
                out[f"{ns}_{col}_{s_}"] = a.mean() if s_ == "mean" else a.std()

        for v in UID_MEAN_STD:
            if has(v):
                agg("uid", uid, v, _num(r[v]))
        for v in C_FOR_MEAN + M_FOR_MEAN:
            if has(v):
                agg("uid", uid, v, _num(r[v]), ("mean",))
        if has("C14"):
            # std of C14 shares the C14 accumulator (already updated above)
            out["uid_C14_std"] = self.agg[("uid", uid, "C14")].std() if uid is not None else float("nan")
        for v in UID_NUNIQUE:
            if has(v):
                if uid is None:
                    out[f"uid_{v}_nunique"] = float("nan"); continue
                val = _key(r[v])
                s_ = self.seen[("nu_" + v, uid)]
                if val is not None:
                    s_.add(val)
                out[f"uid_{v}_nunique"] = float(len(s_))
        for g in ["card1", "card1_addr1", "card1_addr1_P_emaildomain"]:
            if g in keys:
                for v in ["TransactionAmt", "D9", "D11"]:
                    if has(v):
                        agg(g, keys[g], v, _num(r[v]))

        # behaviour (windows, prior-only deviation, lags, novelty)
        hist = self.hist[uid] if uid is not None else None
        n_prior = self.count[("uid", uid)] if uid is not None else float("nan")
        out["uid_tx_count"] = n_prior + 1 if uid is not None else float("nan")

        def uwin(w):
            ww = self._window("uid", uid, w); ww.evict(t)
            return ww

        tag = lambda w: f"{w // 3600}h" if w < 86400 else f"{w // 86400}d"   # noqa: E731
        all_w = set(cfg["velocity_windows_sec"]) | {600, 3600, 86400, 2 * 86400, 7 * 86400}
        wins = {}
        if uid is not None:
            for w in all_w:
                ww = uwin(w); ww.push(t, amt); wins[w] = (float(len(ww.q)), ww.s)
        for w in cfg["velocity_windows_sec"]:
            out[f"uid_n_{tag(w)}"] = wins[w][0] if uid is not None else float("nan")
            out[f"uid_amt_sum_{tag(w)}"] = wins[w][1] if uid is not None else float("nan")

        prior = self.agg[("prior_amt", uid)] if uid is not None else None
        pm = prior.mean() if prior is not None else float("nan")
        ps = prior.std() if prior is not None else float("nan")
        z = (amt - pm) / max(0.0 if math.isnan(ps) else ps, 1.0)
        out["amt_z_vs_own_past"] = float(np.clip(z, -10, 10)) if not math.isnan(z) else float("nan")
        out["amt_ratio_vs_own_past"] = amt / max(pm, 0.01) if not math.isnan(pm) else float("nan")
        if prior is not None:
            prior.add(amt)

        prev = hist[-1] if hist else None
        out["uid_hours_since_prev"] = (t - prev["t"]) / 3600.0 if prev else float("nan")
        out["uid_prev_amt"] = prev["amt"] if prev else float("nan")
        out["uid_amt_diff_prev"] = amt - prev["amt"] if prev else float("nan")
        if has("ProductCD"):
            pc = _key(r["ProductCD"])
            out["uid_same_product_as_prev"] = (float(pc == prev["ProductCD"]) if prev and pc is not None
                                               and prev["ProductCD"] is not None else float("nan"))
        if uid is not None:
            self.first_t.setdefault(("uid", uid), t)
            out["uid_days_since_first"] = (t - self.first_t[("uid", uid)]) / 86400.0
        else:
            out["uid_days_since_first"] = float("nan")
        out["uid_tx_index"] = n_prior if uid is not None else float("nan")
        for v in FIRST_SEEN_VALUES:
            if has(v):
                out[f"uid_new_{v}"] = self._first_time("fs_" + v, uid, _key(r[v]))

        # behaviour v3
        if cfg.get("use_behaviour_v3", True):
            if uid is not None:
                n1h, n7d = wins[3600][0], wins[7 * 86400][0]
                out["uid_n_10m"] = wins[600][0]
                out["uid_n1h_vs_7d_rate"] = n1h / (n7d / 168.0)
                out["uid_amt_1d_vs_prev_1d"] = wins[86400][1] / max(wins[2 * 86400][1] - wins[86400][1], 1.0)
            else:
                for k_ in ["uid_n_10m", "uid_n1h_vs_7d_rate", "uid_amt_1d_vs_prev_1d"]:
                    out[k_] = float("nan")
            for ent in ["card1_addr1", "device_fp"]:
                if ent in keys:
                    ek = keys[ent]
                    for w, tg in [(3600, "1h"), (86400, "1d"), (7 * 86400, "7d")]:
                        if ek is None:
                            out[f"{ent}_n_{tg}"] = out[f"{ent}_amt_sum_{tg}"] = float("nan"); continue
                        ww = self._window("v3_" + ent, ek, w); ww.evict(t); ww.push(t, amt)
                        out[f"{ent}_n_{tg}"] = float(len(ww.q)); out[f"{ent}_amt_sum_{tg}"] = ww.s
            hl_tag = {3600: "1h", 86400: "1d", 7 * 86400: "7d", 30 * 86400: "30d"}
            counts = {}
            for hl in cfg["ema_half_lives_sec"]:
                tg = hl_tag.get(int(hl), f"{int(hl)}s")
                if uid is None:
                    for k_ in [f"uid_ema_n_{tg}", f"uid_ema_amt_mean_{tg}", f"uid_ema_amt_z_{tg}"]:
                        out[k_] = float("nan")
                    counts[hl] = float("nan"); continue
                st = self.ema.get((uid, hl))
                if st is None:
                    st = [0.0, 0.0, 0.0, t]
                d = 0.5 ** ((t - st[3]) / hl)
                c, s1, s2 = st[0] * d, st[1] * d, st[2] * d
                mean = s1 / c if c > 1e-9 else float("nan")
                var = s2 / c - mean * mean if c > 1e-9 else float("nan")
                std = math.sqrt(max(var, 0.0)) if not math.isnan(var) else float("nan")
                out[f"uid_ema_n_{tg}"] = c
                out[f"uid_ema_amt_mean_{tg}"] = mean
                zz = (amt - mean) / max(0.0 if math.isnan(std) else std, 1.0)
                out[f"uid_ema_amt_z_{tg}"] = float(np.clip(zz, -10, 10)) if not math.isnan(zz) else float("nan")
                counts[hl] = c
                self.ema[(uid, hl)] = [c + 1.0, s1 + amt, s2 + amt * amt, t]
            hls = sorted(counts)
            if len(hls) >= 2:
                sh, lo = hls[0], hls[-1]
                out["uid_ema_burst"] = counts[sh] / (counts[lo] * sh / lo + 1e-3)
            if has("hour"):
                if uid is not None:
                    hk = (uid, int(r["hour"]))
                    same = self.count[("uid_hour", hk)]
                    out["uid_hour_typicality"] = same / n_prior if n_prior > 0 else float("nan")
                    self.count[("uid_hour", hk)] += 1
                else:
                    out["uid_hour_typicality"] = float("nan")
            out["uid_history_len"] = n_prior if uid is not None else float("nan")
            for k in (2, 3):
                lag = hist[-k] if hist and len(hist) >= k else None
                out[f"uid_amt_lag{k}"] = lag["amt"] if lag else float("nan")
                out[f"uid_hours_since_lag{k}"] = (t - lag["t"]) / 3600.0 if lag else float("nan")
            for col in ["device_fp", "DeviceInfo", "P_emaildomain", "R_emaildomain", "id_31", "id_30", "dist1"]:
                if has(col):
                    cur = _key(r[col]); pv = prev.get(col) if prev else None
                    out[f"uid_changed_{col}"] = (float(cur != pv) if prev and cur is not None and pv is not None
                                                 else float("nan"))

        # HMM sequence surprise
        if cfg.get("use_hmm_features", True) and self.hmm:
            for key_col, ptag in HMM_PERSPECTIVES:
                if key_col not in keys:
                    continue
                k = keys[key_col]
                prev_t = self.hmm_last_t.get((ptag, k)) if k is not None else None
                xs = {"amt": math.log1p(amt),
                      "gap": math.log1p(max(t - prev_t, 0) / 3600.0) if prev_t is not None else float("nan")}
                for obs, x in xs.items():
                    name = f"{ptag}_{obs}"
                    if name not in self.hmm:
                        continue
                    if k is None:
                        out[f"hmm_{name}_surprise"] = out[f"hmm_{name}_window_ll"] = float("nan"); continue
                    buf = self.hmm_x[(ptag, obs, k)]
                    lags = list(buf)
                    W = np.array([[*([float("nan")] * (2 - len(lags)) + lags), x]], dtype=np.float64)
                    sp, ll = hmm_surprise_from_window(W, self.hmm[name])
                    out[f"hmm_{name}_surprise"] = float(sp[0]); out[f"hmm_{name}_window_ll"] = float(ll[0])
                    buf.append(x)
                if k is not None:
                    self.hmm_last_t[(ptag, k)] = t

        # graph: distinct neighbours so far
        if cfg.get("use_graph_features", True):
            for ent, nb in SHARED_ENTITY_PAIRS:
                if has(ent) and has(nb):
                    ek, nk = _key(r[ent]), _key(r[nb])
                    if ek is None:
                        out[f"{ent}_n_{nb}"] = float("nan"); continue
                    s_ = self.seen[("g_" + ent + "_" + nb, ek)]
                    if nk is not None:
                        s_.add(nk)
                    out[f"{ent}_n_{nb}"] = float(len(s_))

        # network v4 + label history need matured labels first
        def delayed(ns, key):
            if key is None:
                return float("nan"), float("nan"), float("nan")
            d = self.delayed[(ns, key)]
            return d.n, d.f, d.fm

        if cfg.get("use_network_v4", True):
            for ent in ["device_fp", "card1_addr1"]:
                if ent not in keys:
                    continue
                ek = keys[ent]
                pair = (ek, uid) if ek is not None and uid is not None else None
                for w, tg in [(86400, "1d"), (7 * 86400, "7d"), (30 * 86400, "30d")]:
                    if ek is None:
                        out[f"{ent}_clients_{tg}"] = float("nan"); continue
                    ww = self._window("nw_" + ent, ek, w); ww.evict(t); ww.push(t, item=uid)
                    out[f"{ent}_clients_{tg}"] = float(len(ww.cnt))
                is_new = self._first_time("pair_" + ent, ek, uid)
                out[f"{ent}_pair_is_new"] = is_new
                if pair is not None:
                    self.first_t.setdefault(("pair_" + ent, pair), t)
                    out[f"{ent}_pair_age_days"] = (t - self.first_t[("pair_" + ent, pair)]) / 86400.0
                else:
                    out[f"{ent}_pair_age_days"] = float("nan")
                if ek is not None:
                    self.first_t.setdefault(("ent_" + ent, ek), t)
                    out[f"{ent}_age_days"] = (t - self.first_t[("ent_" + ent, ek)]) / 86400.0
                    ww = self._window("nl_" + ent, ek, 86400); ww.evict(t)
                    ww.push(t, 0.0 if math.isnan(is_new) else is_new)
                    out[f"{ent}_new_links_1d"] = ww.s
                else:
                    out[f"{ent}_age_days"] = out[f"{ent}_new_links_1d"] = float("nan")
                en, ef, efm = delayed("lbl_" + ent, ek)
                pn, pf, pfm = delayed("pairlbl_" + ent, pair) if pair is not None else (0.0, 0.0, 0.0)
                on, of, ofm = en - pn, ef - pf, efm - pfm
                out[f"{ent}_other_clients_lbl_fraud"] = of
                out[f"{ent}_other_clients_lbl_rate"] = (ofm + self.smooth * self.prior) / (on + self.smooth)
            # identity clusters (union-find with hub cap, matured labels)
            if "device_fp" in keys:
                dk = keys["device_fp"]
                if uid is None:
                    for k_ in ["cluster_n_clients", "cluster_lbl_fraud", "cluster_lbl_rate", "cluster_merges_identities"]:
                        out[k_] = float("nan")
                else:
                    cn = ("c", uid)
                    rc = self.dsu.find(cn)
                    newp = 0.0
                    if dk is not None:
                        newp = 1.0 if uid not in self.seen[("uf_" + "device_fp", dk)] else 0.0
                    can = dk is not None and self.dsu.deg[dk] + newp <= self.cfg.get("hub_cap", 20)
                    merge = 0.0
                    if can:
                        rl = self.dsu.find(("l", dk))
                        if rl != rc and self.dsu.size[rl] > 0:
                            merge = 1.0
                    out["cluster_n_clients"] = self.dsu.size[rc]
                    out["cluster_lbl_fraud"] = self.dsu.fraud[rc]
                    out["cluster_lbl_rate"] = (self.dsu.fraud_m[rc] + self.smooth * self.prior) / (self.dsu.lab[rc] + self.smooth)
                    out["cluster_merges_identities"] = merge
                    if dk is not None:
                        self.dsu.deg[dk] += newp
                        self.seen[("uf_device_fp", dk)].add(uid)
                    if can:
                        self.dsu.union(cn, ("l", dk))
            if "device_fp" in keys and "card1_addr1" in keys:
                bk = (keys["device_fp"], keys["card1_addr1"])
                if None in bk:
                    out["ring_clients_sharing_device_and_card"] = float("nan")
                else:
                    s_ = self.seen[("ring", bk)]
                    if uid is not None:
                        s_.add(uid)
                    out["ring_clients_sharing_device_and_card"] = float(len(s_)) - 1.0

        if cfg.get("use_label_history", True):
            for k in LABEL_KEYS:
                if k in keys:
                    n, f, fm = delayed("lh_" + k, keys[k])
                    out[f"{k}_lbl_n"], out[f"{k}_lbl_fraud"] = n, f
                    out[f"{k}_lbl_rate"] = (fm + self.smooth * self.prior) / (n + self.smooth) if not math.isnan(n) else float("nan")

        # ── UPDATE shared per-entity state after all reads ───────────────────
        for v in FIRST_SEEN_VALUES:
            if has(v) and uid is not None and _key(r[v]) is not None:
                self.seen[("fs_" + v, uid)].add(_key(r[v]))
        for ent in ["device_fp", "card1_addr1"]:
            if ent in keys and keys[ent] is not None and uid is not None:
                self.seen[("pair_" + ent, keys[ent])].add(uid)
            if ent in keys and keys[ent] is not None:
                self.count[("pair_" + ent, keys[ent])] += 1
        for v in FIRST_SEEN_VALUES:
            if has(v) and uid is not None:
                self.count[("fs_" + v, uid)] += 1
        if uid is not None:
            self.count[("uid", uid)] += 1
            hist.append({"t": t, "amt": amt, **{c: _key(r[c]) for c in
                         ["ProductCD", "device_fp", "DeviceInfo", "P_emaildomain", "R_emaildomain", "id_31", "id_30", "dist1"]
                         if has(c)}})
        txn_keys = {f"lbl_{e}": keys.get(e) for e in ["device_fp", "card1_addr1"]}
        txn_keys.update({f"pairlbl_{e}": (keys[e], uid) for e in ["device_fp", "card1_addr1"]
                         if keys.get(e) is not None and uid is not None})
        txn_keys.update({f"lh_{k}": keys.get(k) for k in LABEL_KEYS if k in keys})
        tid = r["TransactionID"]
        self.txn_keys[tid] = (t, txn_keys, uid)
        self.index_order.append((t, tid))
        self.n_scored += 1
        if labelled:       # every scored txn becomes a (mostly negative) labelled row once mature
            self._seq += 1
            heapq.heappush(self.events, (int(maturity_at(t, self.policy)), self._seq, _EV_MATURE, txn_keys, uid))
        # last-touch order of every entity this event touched (drives TTL pruning)
        tch = self.touched
        for k in [uid, *keys.values()]:
            if k is not None:
                tch[k] = t
                tch.move_to_end(k)
        return out

    # ── bounded state ────────────────────────────────────────────────────────
    def prune(self, now: Optional[int] = None) -> Dict[str, int]:
        """Drop state that can no longer influence any feature (exact) or exceeds its TTL.

        1. windows idle for more than the longest window are empty -> deleted       (exact)
        2. txn index / fraud-id set older than the longest label delay + grace are dropped
           (a label arriving later is reported 'unknown_transaction')                (policy)
        3. decision cache trimmed to its size bound                                (policy)
        4. entities idle for more than profile_ttl_days lose their profile state    (approximate:
           they look new again; ttl >= longest window so in-window features stay exact)
        O(state); called every ``prune_every`` scored transactions, under the engine lock."""
        with self._lock:
            now = self.clock if now is None else int(now)
            before = sum(self.state_size().values())
            # 1. windows
            for k in list(self.win):
                d = self.win[k]
                for w in list(d):
                    d[w].evict(now)
                    if not d[w].q:
                        del d[w]
                if not d:
                    del self.win[k]
            # 2. txn index
            cut = now - self.index_ttl
            while self.index_order and self.index_order[0][0] < cut:
                _, tid = self.index_order.popleft()
                self.txn_keys.pop(tid, None); self.fraud_ids.discard(tid)
            # 3. decision cache
            while len(self.decisions) > self.decision_cache_size:
                self.decisions.popitem(last=False)
            # 4. idle entity profiles
            if self.profile_ttl is not None:
                self._evict_idle_profiles(now - self.profile_ttl)
            sizes = self.state_size()
            return {"before": before, "after": sum(sizes.values()), **sizes}

    def _evict_idle_profiles(self, cutoff: int) -> None:
        tch = self.touched
        stale = set()
        for k, t in tch.items():                      # LRU order: oldest first
            if t >= cutoff:
                break
            stale.add(k)
        if not stale:
            return

        def dead(key) -> bool:
            if isinstance(key, tuple):
                return any(x in stale for x in key)
            return key in stale

        for d in (self.agg, self.seen, self.count, self.delayed, self.win, self.first_t):
            for k in [k for k in d if dead(k[1])]:
                del d[k]
        for k in [k for k in self.ema if k[0] in stale]:
            del self.ema[k]
        for k in [k for k in self.hist if k in stale]:
            del self.hist[k]
        for k in [k for k in self.hmm_x if k[2] in stale]:
            del self.hmm_x[k]
        for k in [k for k in self.hmm_last_t if k[1] in stale]:
            del self.hmm_last_t[k]
        for k in stale:
            del tch[k]

    def state_size(self) -> Dict[str, int]:
        return {"win": len(self.win), "agg": len(self.agg), "seen": len(self.seen), "count": len(self.count),
                "first_t": len(self.first_t), "hist": len(self.hist), "ema": len(self.ema),
                "hmm": len(self.hmm_x) + len(self.hmm_last_t), "labels": len(self.delayed),
                "events": len(self.events), "txn_index": len(self.txn_keys), "decisions": len(self.decisions),
                "touched": len(self.touched), "dsu": len(self.dsu.parent)}

    # ── state persistence (snapshot; Redis-backed store is the production upgrade) ──
    _STATE = ["win", "agg", "seen", "count", "first_t", "hist", "ema", "hmm_x", "hmm_last_t",
              "delayed", "dsu", "events", "browser_newest", "txn_keys", "index_order", "fraud_ids",
              "decisions", "touched", "clock", "n_scored", "label_stats", "_seq"]
    SNAPSHOT_FORMAT = 3

    def save_state(self, path, extra: Optional[Dict] = None) -> None:
        """Atomic snapshot + fingerprint (engine meta/feature list/label policy + caller's ``extra``,
        e.g. the model bundle fingerprint). load_state refuses a snapshot whose fingerprint differs."""
        with self._lock:
            blob = {"_format": self.SNAPSHOT_FORMAT, "_fingerprint": self.fingerprint(), "_extra": extra or {},
                    "state": {k: getattr(self, k) for k in self._STATE}}
            tmp = f"{path}.tmp{os.getpid()}"
            with open(tmp, "wb") as f:
                pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)

    def load_state(self, path, expect_extra: Optional[Dict] = None) -> None:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if not isinstance(blob, dict) or blob.get("_format") != self.SNAPSHOT_FORMAT:
            raise SnapshotMismatchError(f"{path}: unknown snapshot format (pre-validation snapshot?); rebuild it")
        mine = self.fingerprint()
        diff = sorted(k for k in set(mine) | set(blob["_fingerprint"]) if mine.get(k) != blob["_fingerprint"].get(k))
        if expect_extra is not None:
            diff += [f"extra.{k}" for k in sorted(set(expect_extra) | set(blob["_extra"]))
                     if expect_extra.get(k) != blob["_extra"].get(k)]
        if diff:
            raise SnapshotMismatchError(f"{path}: snapshot does not match the loaded model/meta ({', '.join(diff)}); "
                                        "rebuild it by replaying history")
        with self._lock:
            for k, v in blob["state"].items():
                setattr(self, k, v)

    # ── public API ───────────────────────────────────────────────────────────
    def _encode(self, row: pd.Series, stateful: Dict) -> pd.Series:
        vals = {}
        for c in self.feature_cols:
            v = stateful[c] if c in stateful else row.get(c, np.nan)
            if c in self.cat_index:
                v = np.nan if _isnan(v) else self.cat_index[c].get(str(v), np.nan)
            vals[c] = np.nan if _isnan(v) else float(v)
        return pd.Series(vals, dtype=np.float64)

    def check_fields(self, txn: Dict) -> None:
        """Schema=full bundles need Vesta-only fields no real gateway sends: fail loudly, not silently NaN."""
        if self.vendor_fields and self.require_vendor:
            norm = {str(k).replace("id-", "id_") for k in txn}
            missing = [c for c in self.vendor_fields if c not in norm]
            if len(missing) > 0.5 * len(self.vendor_fields):
                raise MissingFieldsError(
                    f"this bundle uses schema=full and needs {len(self.vendor_fields)} Vesta-only fields "
                    f"(V*, C*, D2-D15, id_01-id_11) that the request lacks (e.g. {missing[:5]}). "
                    "Serve the raw-schema bundle (decision_v2.serving_schema: raw).")

    def featurize(self, txn: Dict, labelled: bool = True):
        """-> (row dict, features Series). Thread-safe; raises DuplicateTransactionError for a known id
        (no state change). The expensive row-level preprocessing runs outside the lock."""
        self.check_fields(txn)
        rf = self.row_frame(pd.DataFrame([txn]))
        r = rf.iloc[0].to_dict()
        with self._lock:
            if r["TransactionID"] in self.txn_keys:
                self.label_stats["duplicate_scores"] += 1
                raise DuplicateTransactionError(f"TransactionID {r['TransactionID']} was already scored")
            st = self._stateful(r, labelled)
            x = self._encode(rf.iloc[0], st)
            if self.prune_every and self.n_scored % self.prune_every == 0:
                self.prune()
        return r, x

    def transform(self, txn: Dict) -> pd.Series:
        """Features for ONE new transaction (state is updated: it now counts as history)."""
        return self.featurize(txn)[1]

    # ── decision cache (idempotent re-scoring) ───────────────────────────────
    def cached_decision(self, txn_id):
        with self._lock:
            return self.decisions.get(txn_id)

    def cache_decision(self, txn_id, rec: Dict) -> None:
        with self._lock:
            self.decisions[txn_id] = rec
            while len(self.decisions) > self.decision_cache_size:
                self.decisions.popitem(last=False)

    # ── labels ───────────────────────────────────────────────────────────────
    def record_label(self, txn_id, is_fraud: float, known_at: Optional[int] = None) -> str:
        """A chargeback / confirmed-fraud report. Returns a status string:
          registered          fraud label queued; counts once its (policy) known-time is reached
          already_registered  idempotent repeat
          negative_ignored    is_fraud=0: nothing to do, a mature unreported txn IS a negative
          unknown_transaction never scored here, or older than the label TTL (counted + logged)
          invalid             NaN label
        ``known_at`` (epoch seconds in TransactionDT units) overrides the policy's chargeback date
        (real feed); default = t + fraud_delay(txn_id), which is what offline features assumed."""
        with self._lock:
            if _isnan(is_fraud):
                self.label_stats["invalid"] += 1
                return "invalid"
            if float(is_fraud) != 1.0:
                self.label_stats["negative_ignored"] += 1
                return "negative_ignored"
            ent = self.txn_keys.get(txn_id)
            if ent is None:
                self.label_stats["unknown_transaction"] += 1
                logger.warning("label for unknown/expired TransactionID {} dropped (count={})",
                               txn_id, self.label_stats["unknown_transaction"])
                return "unknown_transaction"
            if txn_id in self.fraud_ids:
                self.label_stats["already_registered"] += 1
                return "already_registered"
            t, keys, uid = ent
            fk = int(fraud_known_at(txn_id, t, self.policy)) if known_at is None else int(known_at)
            fm = max(fk, int(maturity_at(t, self.policy)))
            self._seq += 1
            heapq.heappush(self.events, (fk, self._seq, _EV_FRAUD, keys, uid))
            heapq.heappush(self.events, (fm, self._seq, _EV_FRAUD_MATURE, keys, uid))
            self.fraud_ids.add(txn_id)
            self.label_stats["registered"] += 1
            return "registered"

    def replay(self, raw: pd.DataFrame, labels: bool = True, collect: bool = True,
               batch: int = 50_000) -> Optional[pd.DataFrame]:
        """Run history through the engine in time order (warm-up and parity checks).

        Row-level features are computed vectorised per batch with the offline code; stateful
        features event by event. If labels=True, only FRAUD labels (isFraud == 1) are posted, right
        after the row (a chargeback feed); known-time comes from the label policy. Rows whose isFraud
        is NaN are unlabelled and never count (like offline Kaggle-test rows). Already-scored ids are skipped."""
        raw = raw.sort_values(["TransactionDT", "TransactionID"], kind="mergesort")
        out_rows = []
        for s0 in range(0, len(raw), batch):
            rf = self.row_frame(raw.iloc[s0:s0 + batch].reset_index(drop=True))
            ys = raw["isFraud"].iloc[s0:s0 + batch].to_numpy(dtype=np.float64) if "isFraud" in raw.columns else None
            recs = rf.to_dict("records")
            with self._lock:
                for i, r in enumerate(recs):
                    if r["TransactionID"] in self.txn_keys:
                        continue
                    lab = (ys is None or not np.isnan(ys[i])) if labels else True
                    st = self._stateful(r, labelled=bool(lab))
                    if collect:
                        out_rows.append(self._encode(rf.iloc[i], st))
                    if labels and ys is not None and ys[i] == 1:
                        self.record_label(r["TransactionID"], 1.0)
                    if self.prune_every and self.n_scored % self.prune_every == 0:
                        self.prune()
        return pd.DataFrame(out_rows).reset_index(drop=True) if collect else None

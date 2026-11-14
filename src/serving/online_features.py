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
  * Labels are registered when they arrive (record_label) and only count once they are
    older than label_delay_days — the same rule the training features used.
  * Only agg_mode = causal is servable (transductive needs the future).

Usage
    eng = OnlineFeatureEngine(meta)                  # meta.json from run_feature_engineering
    eng.replay(history_raw_df, labels=True)          # warm state from history (time-sorted)
    x = eng.transform(txn_dict)                      # → pd.Series in meta["feature_cols"] order
    eng.record_label(txn_id, is_fraud)               # when the chargeback / analyst label lands
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
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
from src.feature_engineering.pipeline_v2 import _downcast

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


class _Delayed:
    """Labels of one entity that count only once matured (event_t + delay <= now)."""
    __slots__ = ("pending", "n", "f")

    def __init__(self):
        self.pending: list = []; self.n = 0.0; self.f = 0.0

    def mature(self, now):
        while self.pending and self.pending[0][0] <= now:
            _, y = heapq.heappop(self.pending); self.n += 1; self.f += y


class _DSU:
    def __init__(self):
        self.parent, self.size, self.fraud, self.lab = {}, {}, defaultdict(float), defaultdict(float)
        self.deg = defaultdict(float)

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


class OnlineFeatureEngine:
    def __init__(self, meta: Dict):
        if meta["agg_mode"] != "causal":
            raise ValueError("Only agg_mode=causal features can be served online")
        self.meta = meta
        self.cfg = meta["fe_cfg"]
        self.feature_cols: List[str] = meta["feature_cols"]
        self.delay = int(self.cfg["label_delay_days"] * 86400)
        self.smooth = float(self.cfg["label_smoothing"])
        self.prior = float(meta["train_fraud_rate"])
        self.cat_index = {c: {v: i for i, v in enumerate(cats)} for c, cats in meta["label_maps"].items()}
        self.hmm = {k: tuple(np.asarray(a, dtype=np.float64) for a in v) for k, v in meta.get("hmm_params", {}).items()}
        self._reset_state()

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
        self.delayed = dd(_Delayed)    # (ns, key) -> _Delayed
        self.dsu = _DSU()
        self.dsu_pending: list = []    # heap of (mature_t, seq, client_node, y)
        self.browser_newest: Dict = {}
        self.txn_keys: Dict = {}       # txn_id -> (t, {ns: key}) for record_label
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
    def _stateful(self, r: Dict) -> Dict[str, float]:
        out: Dict[str, float] = {}
        t = int(r["TransactionDT"]); amt = float(r["TransactionAmt"])
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
                      "gap": math.log1p((t - prev_t) / 3600.0) if prev_t is not None else float("nan")}
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
                return float("nan"), float("nan")
            d = self.delayed[(ns, key)]; d.mature(t)
            return d.n, d.f

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
                en, ef = delayed("lbl_" + ent, ek)
                pn, pf = delayed("pairlbl_" + ent, pair) if pair is not None else (0.0, 0.0)
                on, of = en - pn, ef - pf
                out[f"{ent}_other_clients_lbl_fraud"] = of
                out[f"{ent}_other_clients_lbl_rate"] = (of + self.smooth * self.prior) / (on + self.smooth)
            # identity clusters (union-find with hub cap, matured labels)
            if "device_fp" in keys:
                while self.dsu_pending and self.dsu_pending[0][0] <= t:
                    _, _, cnode, y = heapq.heappop(self.dsu_pending)
                    rt = self.dsu.find(cnode); self.dsu.lab[rt] += 1; self.dsu.fraud[rt] += y
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
                    out["cluster_lbl_rate"] = (self.dsu.fraud[rc] + self.smooth * self.prior) / (self.dsu.lab[rc] + self.smooth)
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
                    n, f = delayed("lh_" + k, keys[k])
                    out[f"{k}_lbl_n"], out[f"{k}_lbl_fraud"] = n, f
                    out[f"{k}_lbl_rate"] = (f + self.smooth * self.prior) / (n + self.smooth) if not math.isnan(n) else float("nan")

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
        self.txn_keys[r["TransactionID"]] = (t, txn_keys, uid)
        return out

    # ── state persistence (snapshot; Redis-backed store is the production upgrade) ──
    _STATE = ["win", "agg", "seen", "count", "first_t", "hist", "ema", "hmm_x", "hmm_last_t",
              "delayed", "dsu", "dsu_pending", "browser_newest", "txn_keys", "_seq"]

    def save_state(self, path) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({k: getattr(self, k) for k in self._STATE}, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path) -> None:
        import pickle
        with open(path, "rb") as f:
            for k, v in pickle.load(f).items():
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

    def transform(self, txn: Dict) -> pd.Series:
        """Features for ONE new transaction (state is updated: it now counts as history)."""
        rf = self.row_frame(pd.DataFrame([txn]))
        r = rf.iloc[0].to_dict()
        st = self._stateful(r)
        return self._encode(rf.iloc[0], st)

    def record_label(self, txn_id, is_fraud: float) -> None:
        """Register a label; it is counted only once the transaction is label_delay old."""
        if txn_id not in self.txn_keys or _isnan(is_fraud):
            return
        t, keys, uid = self.txn_keys[txn_id]
        mature = t + self.delay
        for ns, k in keys.items():
            if k is not None:
                heapq.heappush(self.delayed[(ns, k)].pending, (mature, float(is_fraud)))
        if uid is not None:
            self._seq += 1
            heapq.heappush(self.dsu_pending, (mature, self._seq, ("c", uid), float(is_fraud)))

    def replay(self, raw: pd.DataFrame, labels: bool = True, collect: bool = True,
               batch: int = 50_000) -> Optional[pd.DataFrame]:
        """Run history through the engine in time order (warm-up and parity checks).

        Row-level features are computed vectorised per batch with the offline code;
        stateful features event by event. If labels=True, each labelled row's label is
        registered right after the row (it still only counts after the delay)."""
        raw = raw.sort_values(["TransactionDT", "TransactionID"], kind="mergesort")
        out_rows = []
        for s0 in range(0, len(raw), batch):
            rf = self.row_frame(raw.iloc[s0:s0 + batch].reset_index(drop=True))
            ys = raw["isFraud"].iloc[s0:s0 + batch].to_numpy() if (labels and "isFraud" in raw.columns) else None
            recs = rf.to_dict("records")
            for i, r in enumerate(recs):
                st = self._stateful(r)
                if collect:
                    out_rows.append(self._encode(rf.iloc[i], st))
                if ys is not None:
                    self.record_label(r["TransactionID"], ys[i])
        return pd.DataFrame(out_rows).reset_index(drop=True) if collect else None

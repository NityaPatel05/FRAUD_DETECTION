"""
Sequential (event-by-event) feature primitives, compiled with numba.

These mirror how a streaming feature platform (Feedzai Railgun, Redis state) would
compute features online: each row READS the entity's state as it was before the
row, then UPDATES it. Running them over the time-sorted history gives exactly the
values a real-time system would have served — no future data, no own label.

All functions take arrays aligned to the caller's row order (like entity_ops).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from numba import njit

from src.feature_engineering.entity_ops import _Sorted


# ─────────────────────────────────────────────────────────────────────────────
# 1. Exponentially decayed aggregates (Feedzai "light-speed profiling")
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _decayed_prior(c_sorted, t_sorted, x_sorted, half_life):
    n = len(c_sorted)
    out_c = np.full(n, np.nan)
    out_s = np.full(n, np.nan)
    out_s2 = np.full(n, np.nan)
    cnt = 0.0; s = 0.0; s2 = 0.0; last = 0.0
    prev_code = -2
    for i in range(n):
        code = c_sorted[i]
        if code < 0:
            continue
        if code != prev_code:
            cnt = 0.0; s = 0.0; s2 = 0.0; last = t_sorted[i]
            prev_code = code
        d = 0.5 ** ((t_sorted[i] - last) / half_life)
        cnt *= d; s *= d; s2 *= d
        out_c[i] = cnt; out_s[i] = s; out_s2[i] = s2      # READ (prior state)
        x = x_sorted[i]
        if not np.isnan(x):                                 # UPDATE
            cnt += 1.0; s += x; s2 += x * x
        last = t_sorted[i]
    return out_c, out_s, out_s2


def decayed_profile(codes: np.ndarray, t: np.ndarray, x: np.ndarray,
                    half_life_sec: float) -> Dict[str, np.ndarray]:
    """Decayed count / mean / std of ``x`` over the entity's PRIOR rows."""
    s = _Sorted(codes, t)
    c, sm, s2 = _decayed_prior(s.c, s.t.astype(np.float64),
                               np.asarray(x, dtype=np.float64)[s.order], float(half_life_sec))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 1e-9, sm / c, np.nan)
        var = np.where(c > 1e-9, s2 / c - mean * mean, np.nan)
    std = np.sqrt(np.clip(var, 0.0, None))
    return {"count": s.to_orig(c), "mean": s.to_orig(mean), "std": s.to_orig(std)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Exact distinct neighbours inside a sliding window
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _window_distinct(c_sorted, t_sorted, nb_sorted, window, n_nb):
    n = len(c_sorted)
    out = np.full(n, np.nan)
    cnt = np.zeros(n_nb, dtype=np.int64)
    distinct = 0
    lo = 0
    i = 0
    while i < n:
        code = c_sorted[i]
        # block = rows of this entity
        j = i
        while j < n and c_sorted[j] == code:
            j += 1
        if code >= 0:
            lo = i
            distinct = 0
            for k in range(i, j):
                tk = t_sorted[k]
                while lo < k and t_sorted[lo] <= tk - window:   # evict expired rows
                    v = nb_sorted[lo]
                    if v >= 0:
                        cnt[v] -= 1
                        if cnt[v] == 0:
                            distinct -= 1
                    lo += 1
                v = nb_sorted[k]
                if v >= 0:
                    if cnt[v] == 0:
                        distinct += 1
                    cnt[v] += 1
                out[k] = distinct                          # incl. current row
            for k in range(lo, j):                         # reset counters
                v = nb_sorted[k]
                if v >= 0:
                    cnt[v] = 0
        i = j
    return out


def window_distinct(codes: np.ndarray, t: np.ndarray, nb_codes: np.ndarray,
                    window_sec: int) -> np.ndarray:
    """Distinct non-null neighbours of the entity with time in (t - w, t]."""
    s = _Sorted(codes, t)
    nb = np.asarray(nb_codes, dtype=np.int64)[s.order]
    n_nb = int(nb.max()) + 1 if len(nb) and nb.max() >= 0 else 1
    out = _window_distinct(s.c, s.t, nb, np.int64(window_sec), n_nb)
    return s.to_orig(out)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Incremental identity clustering (union-find) with delayed labels
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:                # path compression
        nxt = parent[x]; parent[x] = root; x = nxt
    return root


@njit(cache=True)
def _union_find_stream(t, client, link, link_new_pair, ev_t, ev_row, ev_kind, hub_cap, n_client, n_link):
    """Rows must be in time order. Nodes: clients [0, n_client), links after.

    Label events (sorted by ev_t): kind 0 = row becomes labelled/mature, 1 = row becomes a
    known fraud, 2 = known fraud that is also mature. They enter the client's component
    when ev_t <= the current row time.
    """
    n = len(t)
    ne = len(ev_t)
    parent = np.arange(n_client + n_link)
    size = np.zeros(n_client + n_link)          # number of CLIENT nodes in component
    size[:n_client] = 1.0
    fraud = np.zeros(n_client + n_link)
    fraud_m = np.zeros(n_client + n_link)
    labelled = np.zeros(n_client + n_link)
    deg = np.zeros(n_link)                      # distinct clients per link entity
    out_size = np.full(n, np.nan)
    out_fraud = np.full(n, np.nan)
    out_fraud_m = np.full(n, np.nan)
    out_lab = np.full(n, np.nan)
    out_merge = np.full(n, np.nan)
    p = 0                                       # pointer: next label event to mature
    for i in range(n):
        while p < ne and ev_t[p] <= t[i]:       # labels that are known by now
            r = _find(parent, client[ev_row[p]])
            k = ev_kind[p]
            if k == 0:
                labelled[r] += 1.0
            elif k == 1:
                fraud[r] += 1.0
            else:
                fraud_m[r] += 1.0
            p += 1
        c = client[i]
        if c < 0:
            continue
        rc = _find(parent, c)
        lk = link[i]
        merge = 0.0
        size_now = size[rc]; fraud_now = fraud[rc]; lab_now = labelled[rc]; fm_now = fraud_m[rc]
        can_link = lk >= 0 and deg[lk] + link_new_pair[i] <= hub_cap
        if can_link:
            rl = _find(parent, n_client + lk)
            if rl != rc and size[rl] > 0:
                merge = 1.0                     # bridges two existing identities
        out_size[i] = size_now; out_fraud[i] = fraud_now; out_fraud_m[i] = fm_now
        out_lab[i] = lab_now; out_merge[i] = merge
        if lk >= 0:
            deg[lk] += link_new_pair[i]
        if can_link:                            # UPDATE after reading
            rl = _find(parent, n_client + lk)
            if rl != rc:
                if size[rl] > size[rc]:
                    rl, rc = rc, rl
                parent[rl] = rc
                size[rc] += size[rl]; fraud[rc] += fraud[rl]; labelled[rc] += labelled[rl]
                fraud_m[rc] += fraud_m[rl]
    return out_size, out_fraud, out_lab, out_merge, out_fraud_m


def _run_clusters(t, client_codes, link_codes, link_new_pair, ev_time, ev_row, ev_kind, hub_cap):
    t = np.asarray(t, dtype=np.int64)
    if np.any(np.diff(t) < 0):
        raise ValueError("identity_clusters: rows must be time-sorted")
    client = np.asarray(client_codes, dtype=np.int64)
    link = np.asarray(link_codes, dtype=np.int64)
    n_client = int(client.max()) + 1 if client.max() >= 0 else 1
    n_link = int(link.max()) + 1 if link.max() >= 0 else 1
    newp = np.nan_to_num(np.asarray(link_new_pair, dtype=np.float64), nan=0.0)
    order = np.argsort(ev_time, kind="mergesort")
    size, fraud, lab, merge, fraud_m = _union_find_stream(
        t, client, link, newp, np.asarray(ev_time, dtype=np.float64)[order],
        np.asarray(ev_row, dtype=np.int64)[order], np.asarray(ev_kind, dtype=np.int64)[order],
        float(hub_cap), n_client, n_link)
    return {"size": size, "fraud": fraud, "labelled": lab, "merge": merge, "fraud_mature": fraud_m}


def identity_clusters(t: np.ndarray, client_codes: np.ndarray, link_codes: np.ndarray,
                      link_new_pair: np.ndarray, y: np.ndarray, delay_sec: int,
                      hub_cap: int = 20) -> Dict[str, np.ndarray]:
    """Causal union-find of clients linked through a shared entity (e.g. device).

    A link entity used by more than ``hub_cap`` distinct clients stops linking
    (shared NAT / family / public devices would otherwise merge everyone).
    Fixed-delay labels (every label known ``delay_sec`` after its row); the policy
    version is ``identity_clusters_policy``. Rows must be sorted by time.
    """
    y = np.asarray(y, dtype=np.float64)
    client = np.asarray(client_codes, dtype=np.int64)
    rows = np.nonzero(~np.isnan(y) & (client >= 0))[0]
    fr = rows[y[rows] == 1]
    tt = np.asarray(t, dtype=np.int64)
    ev_time = np.concatenate([tt[rows], tt[fr], tt[fr]]) + np.int64(delay_sec)
    ev_row = np.concatenate([rows, fr, fr])
    ev_kind = np.concatenate([np.zeros(len(rows)), np.ones(len(fr)), np.full(len(fr), 2)]).astype(np.int64)
    return _run_clusters(t, client_codes, link_codes, link_new_pair, ev_time, ev_row, ev_kind, hub_cap)


def identity_clusters_policy(t: np.ndarray, client_codes: np.ndarray, link_codes: np.ndarray,
                             link_new_pair: np.ndarray, events, hub_cap: int = 20) -> Dict[str, np.ndarray]:
    """``identity_clusters`` with per-row label times from ``LabelPolicy`` (label_ready_times)."""
    client = np.asarray(client_codes, dtype=np.int64)
    times, rows, kinds = [], [], []
    for kind, arr in enumerate([events.mature_time, events.fraud_time, events.fraud_mature_time]):
        idx = np.nonzero(np.isfinite(arr) & (client >= 0))[0]
        times.append(np.asarray(arr)[idx]); rows.append(idx); kinds.append(np.full(len(idx), kind))
    return _run_clusters(t, client_codes, link_codes, link_new_pair, np.concatenate(times),
                         np.concatenate(rows), np.concatenate(kinds).astype(np.int64), hub_cap)


def lag_values(codes: np.ndarray, t: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Value on the entity's k-th previous row (NaN if fewer than k prior rows)."""
    s = _Sorted(codes, t)
    v = np.asarray(values, dtype=np.float64)[s.order]
    out = np.full(s.n, np.nan)
    ok = s.rank >= k
    out[ok] = v[np.nonzero(ok)[0] - k]
    return s.to_orig(out)

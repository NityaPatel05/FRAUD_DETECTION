"""
Vectorised, point-in-time-correct primitives for entity (client/card/device) features.

Every function takes arrays aligned to the caller's row order and returns arrays in
the same order. Internally rows are sorted by (entity, time, original position), so
"prior" always means: same entity, earlier in time, ties broken by row order.

Conventions
-----------
* ``codes``  : int64 entity codes; ``-1`` means "entity unknown" (NaN key) and the
               output for those rows is NaN — missing keys must never be pooled into
               one giant "unknown" entity (that creates hub nodes / fake clients).
* ``t``      : int64 timestamps in seconds (TransactionDT).
* "causal"   : uses only rows up to and including the current row (no future rows).
               This is what a real-time system can serve.
* "transductive": uses every row of the entity, past and future (label-free).
               Valid for Kaggle-style batch scoring, NOT servable in real time.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# t (TransactionDT) is < 4e7 seconds; any shift we apply is < 1e8. 1e9 keeps every
# entity in its own disjoint key range so searchsorted never crosses entities.
_KEY_STRIDE = np.int64(1_000_000_000)


# ─────────────────────────────────────────────────────────────────────────────
# Key helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_codes(df: pd.DataFrame, cols: Sequence[str] | str) -> np.ndarray:
    """Integer code per distinct combination of ``cols``; -1 if any part is NaN."""
    if isinstance(cols, str):
        cols = [cols]
    present = [c for c in cols if c in df.columns]
    if len(present) != len(cols):
        missing = set(cols) - set(present)
        raise KeyError(f"make_codes: missing columns {missing}")
    if len(cols) == 1:
        codes, _ = pd.factorize(df[cols[0]], use_na_sentinel=True)
        return codes.astype(np.int64)
    nan_any = df[list(cols)].isna().any(axis=1).to_numpy()
    codes = df.groupby(list(cols), dropna=False, sort=False).ngroup().to_numpy().astype(np.int64)
    codes[nan_any] = -1
    return codes


class _Sorted:
    """Sort rows by (code, t, position) once; reuse for several features."""

    def __init__(self, codes: np.ndarray, t: np.ndarray):
        codes = np.asarray(codes, dtype=np.int64)
        t = np.asarray(t, dtype=np.int64)
        n = len(codes)
        if n and codes.max() >= 9_000_000_000:
            raise OverflowError("entity codes too large for the sort key; factorize them first")
        self.n = n
        self.valid = codes >= 0
        # lexsort: last key is primary → (code, t, position)
        self.order = np.lexsort((np.arange(n), t, codes))
        self.inv = np.empty(n, dtype=np.int64)
        self.inv[self.order] = np.arange(n)
        self.c = codes[self.order]
        self.t = t[self.order]
        self.key = self.c * _KEY_STRIDE + self.t
        # first sorted position of each row's entity block
        self.start = np.searchsorted(self.key, self.c * _KEY_STRIDE, side="left")
        self.pos = np.arange(n)
        # rank within entity (0 = first transaction of the entity)
        self.rank = self.pos - self.start

    def to_orig(self, sorted_vals: np.ndarray) -> np.ndarray:
        out = np.asarray(sorted_vals, dtype=np.float64)[self.inv]
        out[~self.valid] = np.nan
        return out

    def segment_cumsum(self, x_sorted: np.ndarray) -> np.ndarray:
        """Inclusive cumulative sum restarted at every entity block."""
        cs = np.cumsum(x_sorted, dtype=np.float64)
        before = np.where(self.start > 0, cs[np.maximum(self.start - 1, 0)], 0.0)
        return cs - before

    def range_sum(self, cs_global: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Sum of x over sorted positions [lo, hi) given global cumsum of x."""
        upper = np.where(hi > 0, cs_global[np.maximum(hi - 1, 0)], 0.0)
        lower = np.where(lo > 0, cs_global[np.maximum(lo - 1, 0)], 0.0)
        return np.where(hi > lo, upper - lower, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Mean / std / count per entity
# ─────────────────────────────────────────────────────────────────────────────

def entity_mean_std(
    codes: np.ndarray,
    t: np.ndarray,
    values: np.ndarray,
    mode: str = "causal",
    include_current: bool = True,
    s: Optional[_Sorted] = None,
) -> Dict[str, np.ndarray]:
    """Per-entity mean, std (ddof=1) and non-null count of ``values``.

    causal       : over the entity's rows up to (and incl. if include_current) this row.
    transductive : over all of the entity's rows.
    """
    s = s or _Sorted(codes, t)
    v = np.asarray(values, dtype=np.float64)[s.order]
    ok = ~np.isnan(v)
    x = np.where(ok, v, 0.0)

    if mode == "transductive":
        grp = pd.DataFrame({"c": s.c, "x": np.where(ok, v, np.nan)})
        g = grp.groupby("c")["x"]
        mean, std, cnt = (g.transform("mean").to_numpy(), g.transform("std").to_numpy(),
                          g.transform("count").to_numpy().astype(np.float64))
    elif mode == "causal":
        n = s.segment_cumsum(ok.astype(np.float64))
        s1 = s.segment_cumsum(x)
        s2 = s.segment_cumsum(x * x)
        if not include_current:
            n, s1, s2 = n - ok, s1 - x, s2 - x * x
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, s1 / n, np.nan)
            var = np.where(n > 1, (s2 - s1 * s1 / np.maximum(n, 1)) / np.maximum(n - 1, 1), np.nan)
        std = np.sqrt(np.clip(var, 0.0, None))
        cnt = n
    else:
        raise ValueError(f"mode must be 'causal' or 'transductive', got {mode!r}")

    return {"mean": s.to_orig(mean), "std": s.to_orig(std), "count": s.to_orig(cnt)}


def entity_count(codes: np.ndarray, t: np.ndarray, mode: str = "causal",
                 s: Optional[_Sorted] = None) -> np.ndarray:
    """Number of rows of the entity (causal: so far incl. current; transductive: total)."""
    s = s or _Sorted(codes, t)
    if mode == "causal":
        cnt = s.rank + 1.0
    else:
        cnt = pd.Series(s.c).map(pd.Series(s.c).value_counts()).to_numpy().astype(np.float64)
    return s.to_orig(cnt)


# ─────────────────────────────────────────────────────────────────────────────
# Distinct values per entity
# ─────────────────────────────────────────────────────────────────────────────

def entity_nunique(
    codes: np.ndarray,
    t: np.ndarray,
    value_codes: np.ndarray,
    mode: str = "causal",
    s: Optional[_Sorted] = None,
) -> np.ndarray:
    """Number of distinct non-null values seen for the entity.

    ``value_codes`` are int codes of the value column (-1 = NaN, ignored).
    """
    s = s or _Sorted(codes, t)
    vc = np.asarray(value_codes, dtype=np.int64)[s.order]
    ok = vc >= 0
    if mode == "transductive":
        df = pd.DataFrame({"c": s.c, "v": np.where(ok, vc, np.nan)})
        out = df.groupby("c")["v"].transform("nunique").to_numpy().astype(np.float64)
        return s.to_orig(out)
    # causal: a value is "new" the first time the (entity, value) pair appears
    pair = pd.DataFrame({"c": s.c, "v": vc})
    first = (~pair.duplicated(keep="first")).to_numpy() & ok
    return s.to_orig(s.segment_cumsum(first.astype(np.float64)))


def first_time_pair(codes: np.ndarray, t: np.ndarray, value_codes: np.ndarray,
                    s: Optional[_Sorted] = None) -> np.ndarray:
    """1.0 if this is the first time the entity uses this value AND the entity
    has earlier history (a new device/email on an existing client); else 0.0.
    NaN when the value is missing or the entity has no prior row."""
    s = s or _Sorted(codes, t)
    vc = np.asarray(value_codes, dtype=np.int64)[s.order]
    pair = pd.DataFrame({"c": s.c, "v": vc})
    first = (~pair.duplicated(keep="first")).to_numpy().astype(np.float64)
    first[(vc < 0) | (s.rank == 0)] = np.nan
    return s.to_orig(first)


# ─────────────────────────────────────────────────────────────────────────────
# Sliding time windows (velocity) — always causal
# ─────────────────────────────────────────────────────────────────────────────

def window_count_sum(
    codes: np.ndarray,
    t: np.ndarray,
    window_sec: int,
    values: Optional[np.ndarray] = None,
    s: Optional[_Sorted] = None,
) -> Dict[str, np.ndarray]:
    """Count (and optional sum of ``values``) of the entity's rows with
    time in (t - window, t], including the current row. True sliding window."""
    s = s or _Sorted(codes, t)
    lo = np.searchsorted(s.key, s.key - np.int64(window_sec), side="right")
    lo = np.maximum(lo, s.start)
    hi = s.pos + 1
    out = {"count": s.to_orig((hi - lo).astype(np.float64))}
    if values is not None:
        v = np.asarray(values, dtype=np.float64)[s.order]
        cs = np.cumsum(np.nan_to_num(v, nan=0.0))
        out["sum"] = s.to_orig(s.range_sum(cs, lo, hi))
    return out


def prev_row_values(codes: np.ndarray, t: np.ndarray, values: np.ndarray,
                    s: Optional[_Sorted] = None) -> np.ndarray:
    """Value of ``values`` on the entity's previous row (NaN for the first row)."""
    s = s or _Sorted(codes, t)
    v = np.asarray(values, dtype=np.float64)[s.order]
    prev = np.full(s.n, np.nan)
    has_prev = s.rank > 0
    prev[has_prev] = v[np.nonzero(has_prev)[0] - 1]
    return s.to_orig(prev)


# ─────────────────────────────────────────────────────────────────────────────
# Delayed label history (target-based features without leakage)
# ─────────────────────────────────────────────────────────────────────────────

def delayed_label_stats(
    codes: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    delay_sec: int,
    s: Optional[_Sorted] = None,
) -> Dict[str, np.ndarray]:
    """For each row, count the entity's earlier rows whose label is already known
    (row time <= t - delay) and how many of those were fraud.

    ``y`` : 1/0 for labelled rows, NaN for unlabelled rows (e.g. Kaggle test).
    The current row's own label can never be used because delay_sec > 0.
    """
    if delay_sec <= 0:
        raise ValueError("delay_sec must be > 0, otherwise the row's own label leaks")
    s = s or _Sorted(codes, t)
    yv = np.asarray(y, dtype=np.float64)[s.order]
    labelled = (~np.isnan(yv)).astype(np.float64)
    fraud = np.nan_to_num(yv, nan=0.0)
    hi = np.searchsorted(s.key, s.key - np.int64(delay_sec), side="right")
    hi = np.maximum(hi, s.start)
    lo = s.start
    cs_l = np.cumsum(labelled)
    cs_f = np.cumsum(fraud)
    return {
        "n": s.to_orig(s.range_sum(cs_l, lo, hi)),
        "fraud": s.to_orig(s.range_sum(cs_f, lo, hi)),
    }


def smoothed_rate(fraud: np.ndarray, n: np.ndarray, prior: float, strength: float) -> np.ndarray:
    """Bayesian-smoothed fraud rate: (fraud + strength*prior) / (n + strength)."""
    return (fraud + strength * prior) / (n + strength)


def iter_unique(seq: Iterable[str]) -> list:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

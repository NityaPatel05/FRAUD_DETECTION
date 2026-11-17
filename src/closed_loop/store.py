"""What a learning policy knows: the rows it scored, the outcome it could see for each (approved or not, the
propensity the engine logged) and the labels the loop DELIVERED through ``observe``. Never ground truth."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .actions import segment_of
from .triggers import auc_pr
from .weights import ipw_weights

REVIEW_STATUSES = ("review_confirmed", "review_cleared")


class Chunk:
    __slots__ = ("day", "tid", "X", "score", "seg", "approved", "prop", "amount")

    def __init__(self, day, tid, X, score, seg, amount):
        self.day, self.tid, self.X, self.score, self.seg, self.amount = day, tid, X, score.astype(np.float32), seg, amount
        self.approved = np.full(len(tid), -1, dtype=np.int8)       # unknown until the engine's outcome is seen
        self.prop = np.ones(len(tid), dtype=np.float32)


class OutcomeStore:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self._pending: List[Chunk] = []
        self.rep: Dict[int, float] = {}            # transaction_id -> latest delivered reported_label (nan: none)
        self.status: Dict[int, str] = {}

    # ── intake ──
    def add_batch(self, day: int, frame: pd.DataFrame, X: pd.DataFrame, score: np.ndarray) -> None:
        amt = pd.to_numeric(frame["amount_usd"], errors="coerce").astype("float64").to_numpy()
        c = Chunk(day, frame["transaction_id"].to_numpy().astype(np.int64), X.to_numpy(np.float32), np.asarray(score),
                  segment_of(frame), amt)
        self.chunks.append(c)
        self._pending.append(c)

    def ingest_outcomes(self, history) -> None:
        """The decision outcome of the previous epoch(s): approved flag and logged propensity (public columns)."""
        if not self._pending or not history.frames:
            return
        h = history.frames[-1]
        tid = h["transaction_id"].to_numpy().astype(np.int64)
        app, prop = h["approved"].to_numpy(), pd.to_numeric(h["propensity"], errors="coerce").to_numpy()
        keep = []
        for c in self._pending:
            idx = np.searchsorted(tid, c.tid) if (np.diff(tid) > 0).all() else None
            if idx is not None:
                idx = np.clip(idx, 0, len(tid) - 1)
                ok = tid[idx] == c.tid
            else:
                s = pd.Series(np.arange(len(tid)), index=tid)
                pos = s.reindex(c.tid).to_numpy()
                ok = ~np.isnan(pos)
                idx = np.where(ok, pos, 0).astype(int)
            if ok.any():
                c.approved[ok] = app[idx][ok].astype(np.int8)
                c.prop[ok] = np.where(np.isfinite(prop[idx][ok]), prop[idx][ok], 1.0)
            if not ok.all():
                keep.append(c)
        self._pending = keep

    def observe(self, labels: pd.DataFrame) -> None:
        rep = pd.to_numeric(labels["reported_label"], errors="coerce").astype("float64").to_numpy()
        for t, r, s in zip(labels["transaction_id"].to_numpy(), rep, labels["label_status"].to_numpy()):
            self.rep[int(t)] = r
            self.status[int(t)] = s

    # ── training set ──
    def training_set(self, cutoff_day: int, gap_days: int, train_on_all: bool = False, first_day: int = 0,
                     last_day: int | None = None, use_ipw: bool = True, max_weight: float = 100.0):
        """Rows decided in [first_day, min(last_day, cutoff-gap)] whose OUTCOME is observable, with the label as
        delivered so far (pending rows are negatives, exactly like a production system). Returns
        (X, y, w, amount, seg, tid)."""
        hi = cutoff_day - gap_days if last_day is None else min(last_day, cutoff_day - gap_days)
        cs = [c for c in self.chunks if first_day <= c.day <= hi and (c.approved >= 0).any()]
        if not cs:
            return None
        X = np.concatenate([c.X for c in cs])
        tid = np.concatenate([c.tid for c in cs])
        app = np.concatenate([c.approved for c in cs])
        prop = np.concatenate([c.prop for c in cs])
        amt = np.concatenate([c.amount for c in cs])
        seg = np.concatenate([c.seg for c in cs])
        rep = np.fromiter((self.rep.get(int(t), np.nan) for t in tid), dtype=float, count=len(tid))
        reviewed = np.fromiter((self.status.get(int(t), "") in REVIEW_STATUSES for t in tid), dtype=bool, count=len(tid))
        known = app >= 0
        inc = known & ((app == 1) | reviewed | train_on_all)
        y = (np.nan_to_num(rep, nan=0.0) == 1).astype(np.int8)
        w = ipw_weights(prop, max_weight) if use_ipw else np.ones(len(prop))
        w = np.where(reviewed & (app == 0), 1.0, w)
        sel = inc
        return X[sel], y[sel], w[sel].astype(np.float64), amt[sel], seg[sel], tid[sel]

    # ── Monitor interface for triggers ──
    def _in(self, d0, d1):
        return [c for c in self.chunks if d0 <= c.day < d1]

    def scores(self, d0: int, d1: int) -> np.ndarray:
        cs = self._in(d0, d1)
        return np.concatenate([c.score for c in cs]) if cs else np.array([])

    def feature(self, name: str, d0: int, d1: int, cols: List[str] | None = None) -> np.ndarray:
        cs = self._in(d0, d1)
        if not cs or self.cols is None or name not in self.cols:
            return np.array([])
        j = self.cols.index(name)
        return np.concatenate([c.X[:, j] for c in cs])

    cols: List[str] | None = None

    def matured_perf(self, d0: int, d1: int) -> Tuple[float, int]:
        cs = [c for c in self._in(d0, d1) if (c.approved >= 0).any()]
        if not cs:
            return float("nan"), 0
        tid = np.concatenate([c.tid for c in cs])
        sc = np.concatenate([c.score for c in cs])
        app = np.concatenate([c.approved for c in cs])
        m = app == 1
        y = np.array([self.rep.get(int(t), 0.0) == 1 for t in tid[m]], dtype=int)
        return auc_pr(y, sc[m]), int(y.sum())

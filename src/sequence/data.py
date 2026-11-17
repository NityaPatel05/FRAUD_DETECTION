"""Feature encoding for the sequence model: pre-authorization fields only, no labels.

A transaction is described by fields a gateway has when the request arrives (amount, time, product,
card type, email domain, device, address). Everything is fitted on the PRETRAIN window only.
Gap-to-previous-transaction is NOT stored in the static arrays: it depends on client state, so the
batch path (`gaps`) and the online path (`ClientState.last_t`) share `gap_inputs`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

DEFAULT_CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType", "addr1", "addr2", "DeviceInfo"]
N_NUM = 5            # log amt, sin/cos hour, sin/cos weekday
MAX_VOCAB = 60


@dataclass
class Encoded:
    cats: np.ndarray      # [N, C] int64 (0 = missing, 1 = other/unseen, 2.. = known)
    num: np.ndarray       # [N, N_NUM] float32
    amt_b: np.ndarray     # [N] int64 amount bucket
    t: np.ndarray         # [N] int64 seconds

    def __len__(self):
        return len(self.t)

    def row(self, i: int) -> "Encoded":
        return Encoded(self.cats[i:i + 1], self.num[i:i + 1], self.amt_b[i:i + 1], self.t[i:i + 1])


def client_order(client: np.ndarray, t: np.ndarray, tid: np.ndarray):
    """Sort order (client, time, id) with every client<0 row its own singleton client.
    -> (order, client_eff[order], start offsets of each client run)."""
    client = np.asarray(client, dtype=np.int64)
    eff = np.where(client >= 0, client, client.max(initial=0) + 1 + np.arange(len(client)))
    order = np.lexsort((np.asarray(tid), np.asarray(t), eff))
    ce = eff[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ce)) + 1]
    return order, ce, starts


def gaps(t_sorted: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Seconds since the client's previous txn, -1 for a client's first txn (arrays in client order)."""
    g = np.r_[-1, np.diff(t_sorted)].astype(np.float64)
    g[starts] = -1
    return g


def gap_inputs(gap: np.ndarray) -> np.ndarray:
    """[N, 2] numeric gap inputs: log1p(gap) (0 if first) and a first-txn flag."""
    gap = np.asarray(gap, dtype=np.float64)
    first = gap < 0
    return np.stack([np.where(first, 0.0, np.log1p(np.maximum(gap, 0))) / 8.0, first.astype(np.float64)], 1).astype(np.float32)


@dataclass
class SeqSpec:
    cat_cols: List[str]
    vocabs: Dict[str, Dict[object, int]]
    amt_edges: np.ndarray
    gap_edges: np.ndarray
    amt_col: str = "TransactionAmt"
    t_col: str = "TransactionDT"

    @property
    def n_amt(self) -> int:
        return len(self.amt_edges) + 1

    @property
    def n_gap(self) -> int:             # bucket 0 = first transaction of the client
        return len(self.gap_edges) + 2

    @property
    def cat_sizes(self) -> List[int]:
        return [len(self.vocabs[c]) + 2 for c in self.cat_cols]

    @classmethod
    def fit(cls, df: pd.DataFrame, gap: np.ndarray, cat_cols: Sequence[str] = DEFAULT_CAT_COLS,
            n_amt: int = 16, n_gap: int = 12, amt_col: str = "TransactionAmt") -> "SeqSpec":
        cols = [c for c in cat_cols if c in df.columns]
        vocabs = {}
        for c in cols:
            vc = df[c].value_counts().head(MAX_VOCAB)
            vocabs[c] = {v: i + 2 for i, v in enumerate(vc.index)}
        la = np.log1p(df[amt_col].to_numpy(dtype=np.float64))
        ae = np.unique(np.nanquantile(la, np.linspace(0, 1, n_amt + 1)[1:-1]))
        g = gap[gap >= 0]
        ge = np.unique(np.quantile(np.log1p(g), np.linspace(0, 1, n_gap + 1)[1:-1])) if len(g) else np.array([0.0])
        return cls(cols, vocabs, ae, ge, amt_col)

    def encode(self, df: pd.DataFrame) -> Encoded:
        n = len(df)
        cats = np.zeros((n, len(self.cat_cols)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols):
            s = df[c]
            m = s.map(self.vocabs[c]).fillna(1).astype(np.int64).to_numpy()
            cats[:, j] = np.where(s.isna().to_numpy(), 0, m)
        t = df[self.t_col].to_numpy(dtype=np.int64)
        la = np.log1p(df[self.amt_col].to_numpy(dtype=np.float64))
        hr = (t // 3600) % 24
        wd = (t // 86400) % 7
        num = np.stack([la / 6.0, np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
                        np.sin(2 * np.pi * wd / 7), np.cos(2 * np.pi * wd / 7)], 1).astype(np.float32)
        return Encoded(cats, num, np.searchsorted(self.amt_edges, la).astype(np.int64), t)

    def gap_bucket(self, gap: np.ndarray) -> np.ndarray:
        gap = np.asarray(gap, dtype=np.float64)
        b = np.searchsorted(self.gap_edges, np.log1p(np.maximum(gap, 0))) + 1
        return np.where(gap < 0, 0, b).astype(np.int64)

    def encode_one(self, txn: Dict) -> Encoded:
        """Raw field dict (as a gateway request) -> Encoded of length 1."""
        return self.encode(pd.DataFrame([txn]))

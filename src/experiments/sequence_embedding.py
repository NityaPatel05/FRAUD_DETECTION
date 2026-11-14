"""
Phase 9 (optional) — learned sequence score as a GBDT feature, evaluated honestly.

Industry pattern (Feedzai GRU, Amex LSTM+GBM, Nubank nuFormer late fusion): a
UNIDIRECTIONAL recurrent model over the client's recent history whose output is fed to
the tree model — never a bidirectional model (it would read the future) and never a
replacement for the GBDT. Expected gain at this data scale: 0 to +0.01 AUC-PR.

Leakage control: "time-sliced out-of-fold". The timeline is cut into blocks; the score
of every row in block b comes from a GRU trained only on blocks < b (the first block
gets NaN). Val/test rows are scored by a GRU trained on the whole train period.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from src.feature_engineering.entity_ops import _Sorted

STEP_FEATURES = ["log_amt", "log_gap_h", "hour_sin", "hour_cos", "product"]


def step_matrix(df: pd.DataFrame) -> np.ndarray:
    """Per-row step features [n, d] (computable for a single live transaction too)."""
    amt = np.log1p(df["TransactionAmt"].to_numpy(dtype=np.float64))
    hour = df["hour"].to_numpy(dtype=np.float64) if "hour" in df.columns else (df["TransactionDT"].to_numpy() // 3600) % 24
    prod = np.nan_to_num(df["ProductCD"].to_numpy(dtype=np.float64), nan=-1.0) if "ProductCD" in df.columns else np.zeros(len(df))
    gap = np.zeros(len(df))                                # filled per sequence below
    return np.column_stack([amt, gap, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), prod / 5.0])


def build_sequences(df: pd.DataFrame, k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """X [n, k+1, d]: the client's k PREVIOUS transactions + the current one (left-padded),
    mask [n, k+1]. Rows must carry uid1_code and TransactionDT. Strictly causal."""
    codes = df["uid1_code"].to_numpy(dtype=np.int64)
    t = df["TransactionDT"].to_numpy(dtype=np.int64)
    s = _Sorted(codes, t)
    base = step_matrix(df)[s.order]                       # sorted by (client, time)
    ts = t[s.order]
    n, d = base.shape
    X = np.zeros((n, k + 1, d), dtype=np.float32)
    M = np.zeros((n, k + 1), dtype=np.float32)
    for lag in range(k + 1):                             # slot k = current row, slot k-lag = lag-th previous
        ok = (s.rank >= lag) & (s.c >= 0) if lag > 0 else np.ones(n, dtype=bool)
        src = np.nonzero(ok)[0]
        rows = src - lag
        feat = base[rows].copy()
        has_prev = (s.rank[rows] > 0) & (s.c[rows] >= 0)      # unknown clients have no history
        prev_t = np.where(has_prev, ts[np.maximum(rows - 1, 0)], ts[rows])
        feat[:, 1] = np.log1p((ts[rows] - prev_t) / 3600.0)
        X[src, k - lag] = feat
        M[src, k - lag] = 1.0
    inv = s.inv
    return X[inv], M[inv]


def _model(d: int, hidden: int = 32):
    import torch
    import torch.nn as nn

    class SeqGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(d, hidden, batch_first=True)          # unidirectional on purpose
            self.head = nn.Sequential(nn.Linear(hidden + d, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x, m):
            h, _ = self.gru(x * m.unsqueeze(-1))
            last = h[:, -1]                                          # state after the current txn
            return self.head(torch.cat([last, x[:, -1]], dim=1)).squeeze(1)
    return SeqGRU()


def train_score(X_tr, M_tr, y_tr, X_sc, M_sc, epochs: int = 3, batch: int = 4096,
                device: str = "cpu", seed: int = 0) -> np.ndarray:
    """Train on (X_tr, y_tr), return fraud scores for X_sc. Inputs standardised on train."""
    import torch
    torch.manual_seed(seed)
    mu = (X_tr * M_tr[..., None]).sum((0, 1)) / max(M_tr.sum(), 1)
    sd = np.sqrt((((X_tr - mu) ** 2) * M_tr[..., None]).sum((0, 1)) / max(M_tr.sum(), 1)) + 1e-6
    norm = lambda X: np.clip((X - mu) / sd, -5, 5).astype(np.float32)     # noqa: E731
    dev = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
    net = _model(X_tr.shape[2]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    pos_w = torch.tensor([min((1 - y_tr.mean()) / max(y_tr.mean(), 1e-6), 5.0)], device=dev)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt, Mt, yt = torch.from_numpy(norm(X_tr)), torch.from_numpy(M_tr), torch.from_numpy(y_tr.astype(np.float32))
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(Xt[idx].to(dev), Mt[idx].to(dev)), yt[idx].to(dev))
            loss.backward(); opt.step()
    net.eval(); out = []
    with torch.no_grad():
        Xs, Ms = torch.from_numpy(norm(X_sc)), torch.from_numpy(M_sc)
        for i in range(0, len(Xs), batch * 4):
            out.append(torch.sigmoid(net(Xs[i:i + batch * 4].to(dev), Ms[i:i + batch * 4].to(dev))).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def time_sliced_oof(X, M, y, day, block_days: int = 15, **kw) -> np.ndarray:
    """Score each time block with a GRU trained only on EARLIER blocks (first block → NaN)."""
    out = np.full(len(y), np.nan)
    start = int(day.min())
    edges = list(range(start + block_days, int(day.max()) + 1, block_days)) + [int(day.max()) + 1]
    for lo, hi in zip(edges[:-1], edges[1:]):
        tr, sc = day < lo, (day >= lo) & (day < hi)
        if tr.sum() and sc.sum():
            out[sc] = train_score(X[tr], M[tr], y[tr], X[sc], M[sc], **kw)
    return out

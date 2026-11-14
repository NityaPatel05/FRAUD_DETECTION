"""Internal transaction representation: entity *positions* + raw values, before assembly."""
from __future__ import annotations

import numpy as np
import pandas as pd

# column -> (dtype, default)
CORE = {
    "t": (np.int64, None),              # seconds since simulation start (UTC)
    "customer": (np.int64, -1),         # cardholder / account holder (sender for P2P)
    "card": (np.int64, -1),
    "merchant": (np.int64, -1),
    "terminal": (np.int64, -1),
    "counterparty": (np.int64, -1),     # P2P receiver
    "channel": (object, None),          # CP / CNP / P2P
    "pos_entry_mode": (object, None),
    "value_usd": (np.float64, np.nan),  # amount before currency conversion
    "device": (np.int64, -1),
    "ip": (np.int64, -1),
    "email_domain": (object, None),
    "shipping_country": (np.int64, -1),
    "cred_type": (object, None),        # what the payer actually holds (drives AVS/CVC/3DS)
    "is_fraud": (np.int8, 0),
    "scenario": (object, "legit"),
    "incident": (np.int64, -1),
    "ring": (np.int64, -1),
    "delay_days": (np.float64, np.nan),  # label delay; set for fraud rows
}


class Rows:
    """Accumulates chunks of transactions; scalars broadcast to the chunk length."""

    def __init__(self, horizon: int | None = None):
        self.parts: list[dict] = []
        self.horizon = horizon  # if set, rows outside [0, horizon) are dropped on add

    def add(self, n: int, **cols) -> int:
        """Add a chunk; returns the number of rows kept."""
        if n <= 0:
            return 0
        unknown = set(cols) - set(CORE)
        if unknown:
            raise KeyError(f"unknown row columns {unknown}")
        if "t" not in cols:
            raise KeyError("t is required")
        chunk = {}
        for k, (dtype, default) in CORE.items():
            v = cols.get(k, default)
            v = np.asarray(v)
            if v.ndim == 0:
                v = np.full(n, v.item() if v.dtype != object else v[()], dtype=dtype)
            elif len(v) != n:
                raise ValueError(f"{k}: length {len(v)} != {n}")
            chunk[k] = v.astype(dtype, copy=False)
        if self.horizon is not None:
            keep = (chunk["t"] >= 0) & (chunk["t"] < self.horizon)
            if not keep.all():
                chunk = {k: v[keep] for k, v in chunk.items()}
        kept = len(chunk["t"])
        if kept:
            self.parts.append(chunk)
        return kept

    def add_frame(self, d: dict) -> None:
        n = len(d["t"])
        self.add(n, **d)

    def to_frame(self) -> pd.DataFrame:
        if not self.parts:
            return pd.DataFrame({k: np.zeros(0, dtype=dt) for k, (dt, _) in CORE.items()})
        return pd.DataFrame({k: np.concatenate([p[k] for p in self.parts]) for k in CORE})

"""
Segment-level attack detection (Uber RADAR / Stripe card-testing style).

Build a time series per segment value (e.g. ProductCD=C, card6=credit, amount band
50-100, email class=privacy …) and flag periods whose volume or amount is far above
the segment's own recent baseline. Robust baseline = rolling median and MAD of the
PREVIOUS periods only (no look-ahead), so it works in real time.

An alert needs: z >= z_threshold, sustained for >= min_periods consecutive periods,
and volume >= min_volume (avoids noise on tiny segments). Flash-sale style volume
spikes are a known false positive — inspect the mined rules (rule_mining.py) and
the matured fraud rate before enforcing anything.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

AMOUNT_BANDS = [0, 10, 25, 50, 100, 200, 500, 1000, np.inf]


def add_amount_band(df: pd.DataFrame, col: str = "TransactionAmt") -> pd.Series:
    return pd.cut(df[col], AMOUNT_BANDS, right=False).astype(str)


def segment_series(df: pd.DataFrame, segment_cols: Sequence[str], period_sec: int = 3600,
                   time_col: str = "TransactionDT", amount_col: str = "TransactionAmt") -> pd.DataFrame:
    """Long table: [segment, value, period, n, amount] with zero-filled empty periods."""
    period = (df[time_col] // period_sec).astype(np.int64)
    all_periods = np.arange(period.min(), period.max() + 1)
    frames = []
    for col in segment_cols:
        g = (pd.DataFrame({"value": df[col].astype(str), "period": period, "amt": df[amount_col]})
             .groupby(["value", "period"]).agg(n=("amt", "size"), amount=("amt", "sum")).reset_index())
        full = (g.set_index(["value", "period"])
                 .reindex(pd.MultiIndex.from_product([g["value"].unique(), all_periods],
                                                     names=["value", "period"]), fill_value=0)
                 .reset_index())
        full.insert(0, "segment", col)
        frames.append(full)
    return pd.concat(frames, ignore_index=True)


def robust_z(series: pd.DataFrame, metric: str = "n", baseline_periods: int = 168,
             min_baseline: int = 24) -> pd.Series:
    """z = (x - median(prev)) / (1.4826 * MAD(prev)), previous periods only."""
    def _z(s: pd.Series) -> pd.Series:
        prev = s.shift(1)
        med = prev.rolling(baseline_periods, min_periods=min_baseline).median()
        mad = (prev - med).abs().rolling(baseline_periods, min_periods=min_baseline).median()
        scale = 1.4826 * mad
        scale = scale.where(scale > 0, np.sqrt(med.clip(lower=1)))   # Poisson-ish floor for sparse series
        return (s - med) / scale
    return series.groupby(["segment", "value"], sort=False)[metric].transform(_z)


def detect_attacks(df: pd.DataFrame, segment_cols: Sequence[str], period_sec: int = 3600,
                   z_threshold: float = 6.0, min_periods: int = 2, min_volume: int = 20,
                   baseline_periods: int = 168) -> pd.DataFrame:
    """Return one row per alert: segment, value, start/end period, peak z, volume."""
    ser = segment_series(df, segment_cols, period_sec)
    ser["z"] = robust_z(ser, "n", baseline_periods)
    ser["hot"] = (ser["z"] >= z_threshold) & (ser["n"] >= min_volume)
    alerts: List[Dict] = []
    for (seg, val), g in ser.groupby(["segment", "value"], sort=False):
        hot = g["hot"].to_numpy()
        if not hot.any():
            continue
        periods = g["period"].to_numpy(); z = g["z"].to_numpy(); n = g["n"].to_numpy()
        i = 0
        while i < len(hot):
            if hot[i]:
                j = i
                while j + 1 < len(hot) and hot[j + 1]:
                    j += 1
                if j - i + 1 >= min_periods:
                    alerts.append({"segment": seg, "value": val,
                                   "start_period": int(periods[i]), "end_period": int(periods[j]),
                                   "start_time": int(periods[i]) * period_sec,
                                   "end_time": (int(periods[j]) + 1) * period_sec,
                                   "peak_z": float(np.nanmax(z[i:j + 1])), "volume": int(n[i:j + 1].sum())})
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(alerts).sort_values("peak_z", ascending=False).reset_index(drop=True) \
        if alerts else pd.DataFrame(columns=["segment", "value", "start_period", "end_period",
                                             "start_time", "end_time", "peak_z", "volume"])

"""
Calendar effects (all vectors are indexed by simulation day and have mean 1).

* weekly cycle of the POPULATION (Fri/Sat peak, Sunday trough);
* paydays (1st, 15th, last working day), month-end bills;
* holiday peaks (Valentine's, Prime-Day-style July sale, Black Friday / Cyber Monday,
  Christmas run-up and the quiet 25th/1st);
* fraud campaign waves: a self-exciting (Hawkes-like) clustering of attack starts;
* gradual drift of the fraud mix (card testing fades, takeover and synthetic ids grow).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEKLY = np.array([0.95, 0.95, 0.97, 1.00, 1.10, 1.13, 0.90])   # Monday .. Sunday

# scenario -> log-slope of its daily intensity across the horizon (drift of the fraud mix)
MIX_SLOPE = {"card_testing": -0.5, "skimming": -0.6, "terminal_cashout": -0.3, "account_takeover": 0.7,
             "synthetic_identity": 0.5, "merchant_collusion": 0.3, "transaction_laundering": 0.2}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    d += pd.Timedelta(days=(weekday - d.weekday()) % 7)
    return d + pd.Timedelta(days=7 * (n - 1))


def dates(start_date: str, n_days: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start_date), periods=n_days, freq="D")


def day_weights(start_date: str, n_days: int, enabled: bool = True) -> np.ndarray:
    """Legit demand multiplier per day (mean 1)."""
    if not enabled:
        return np.ones(n_days)
    d = dates(start_date, n_days)
    w = WEEKLY[d.dayofweek.to_numpy()].copy()
    dom = d.day.to_numpy()
    dim = d.days_in_month.to_numpy()
    payday = (dom == 1) | (dom == 2) | (dom == 15) | (dom == 16) | (dom == dim) | (dom == dim - 1)
    w *= np.where(payday, 1.16, 1.0)
    # month-end bills / rent window: slightly more activity in the last 3 days
    w *= np.where(dom >= dim - 2, 1.05, 1.0)
    hol = np.ones(n_days)
    for y in range(d[0].year, d[-1].year + 1):
        bf = _nth_weekday(y, 11, 3, 4) + pd.Timedelta(days=1)         # day after 4th Thursday of Nov
        events = [
            (pd.Timestamp(y, 2, 14), 1.10, 2),                       # Valentine's
            (pd.Timestamp(y, 7, 15), 1.20, 2),                       # summer sale
            (bf, 1.70, 1), (bf + pd.Timedelta(days=3), 1.35, 1),    # Black Friday, Cyber Monday
            (pd.Timestamp(y, 12, 22), 1.40, 7),                      # Christmas run-up (decays before)
            (pd.Timestamp(y, 12, 25), 0.55, 0), (pd.Timestamp(y, 1, 1), 0.80, 0),
            (pd.Timestamp(y, 12, 26), 1.20, 1),                      # boxing day sales
        ]
        for day, amp, spread in events:
            for k in range(-spread, spread + 1):
                idx = int((day + pd.Timedelta(days=k) - d[0]).days)
                if 0 <= idx < n_days:
                    hol[idx] *= 1.0 + (amp - 1.0) * (1.0 - abs(k) / (spread + 1.0)) if amp > 1 else amp
    w *= hol
    return w / w.mean()


def fraud_day_weights(rng: np.random.Generator, legit_w: np.ndarray, waves: bool = True) -> np.ndarray:
    """Fraud intensity per day: follows holiday demand (sqrt) and clusters in campaigns."""
    n = len(legit_w)
    w = np.sqrt(legit_w)
    if waves:
        mu, branching, decay = 0.04, 0.55, 9.0          # campaign starts / day, offspring, days
        lam = np.zeros(n + 60)
        excit = 0.0
        for i in range(n):
            lam_i = mu + excit
            k = rng.poisson(lam_i)
            if k:
                size = rng.lognormal(0.0, 0.5, k).sum()
                excit += branching * size / decay
            lam[i] = lam_i
            excit *= np.exp(-1.0 / decay)
        w = w * (0.55 + lam[:n] / max(lam[:n].mean(), 1e-9) * 0.45)
    return w / w.mean()


def mix_weights(n_days: int, scenario: str, enabled: bool = True) -> np.ndarray:
    if not enabled:
        return np.ones(n_days)
    slope = MIX_SLOPE.get(scenario, 0.0)
    x = np.linspace(-0.5, 0.5, n_days)
    return np.exp(slope * x)


def month_table(start_date: str, n_days: int):
    """Day index of each month start inside the horizon (plus one beyond) and month lengths."""
    d = dates(start_date, n_days)
    first = d[0].replace(day=1)
    months = pd.date_range(first, d[-1] + pd.offsets.MonthBegin(1), freq="MS")
    starts = np.array([(m - d[0]).days for m in months], dtype=np.int64)
    lens = np.array([m.days_in_month for m in months], dtype=np.int64)
    return starts, lens

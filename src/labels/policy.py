"""
ONE label policy for the whole system: features, training targets, evaluation, serving.

Real fraud labels are late and incomplete. A chargeback arrives days to months after the
transaction, and a transaction nobody disputed is only *assumed* legitimate after a
maturity window. Every consumer in this repo must see labels exactly as a production
system would have seen them, so they all call this module and nothing else.

Semantics
---------
For a transaction ``i`` with time ``t_i`` (seconds) and id ``id_i``:

* ``d_i = fraud_delay_sec(id_i)``  -- when its FRAUD label becomes known, relative to
  ``t_i``. Drawn from a lognormal (default median 35 d, sigma 0.6, clipped to
  [1 d, 120 d]) with a hash of the TransactionID as the random source: the same id gets
  the same delay in every process (offline training, online serving, tests). The
  delay only matters if the transaction really is fraud.
* A fraud is *known* at time ``now``  iff  ``t_i + d_i <= now``.
* A transaction is *mature* at ``now`` iff ``t_i + maturity_days <= now`` (default 90 d).
  A mature transaction with no known fraud label is treated as LEGIT (a label of 0).
  A fraud whose delay exceeds the maturity window is therefore first seen as legit and
  flips to fraud when its chargeback finally arrives (this is the real-world noise).
* Everything younger than maturity and not known fraud is PENDING (label unknown).

What the label view at time ``now`` looks like (``known_label``):
    1.0  known fraud            (t + d <= now)
    0.0  mature, not known fraud
    NaN  pending

How features count labelled rows (label_history, network and cluster features)
    n      = # earlier rows of the entity that are MATURE                (ready at t+maturity)
    fraud  = # earlier rows of the entity that are KNOWN FRAUD, any age  (ready at t+d)
    rate   = (fraud among MATURE rows + s*prior) / (n + s)
             -> numerator and denominator cover the same population, so the rate is not
                inflated by "frauds are reported early, legit rows only settle late".
    The fresh signal (a card chargebacked last week) lives in ``fraud`` (and
    fraud - fraud_mature = recent known fraud); the stable signal lives in ``rate``.

Training targets as known at a cutoff (``training_target``)
    target_i = 1 iff fraud_i and its label is known by ``cutoff``, else 0 -- a fraud still
    in its chargeback delay is (wrongly) a negative, exactly like in production. Rows in
    the last ``train_gap_days`` before the cutoff are dropped (Fraud Detection Handbook
    "delay period"). Folds are  train | gap | val.

Evaluation uses the TRUE label (isFraud); the policy only restricts what the model and
its features may use.

Online API (pure functions, numpy-scalar or array in/out; no state)
    fraud_delay_sec(txn_id, policy)            seconds until the fraud label (if any) is known
    fraud_known_at(txn_id, t, policy)          absolute time t + d
    fraud_known_by(txn_id, t, now, policy)     bool
    is_mature(t, now, policy)                  bool
    maturity_at(t, policy)                     t + maturity
    known_label(y, txn_id, t, now, policy)     1.0 / 0.0 / NaN as above
    label_ready_times(y, txn_id, t, policy)    the three event times the feature code uses
    training_target(y, txn_id, t, cutoff, policy)   -> (target, usable)

An online engine registers a label with ``fraud_known_at`` as its due time and counts a
row as mature at ``maturity_at`` -- identical to the vectorised offline path.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Dict, Mapping, NamedTuple, Optional, Tuple, Union

import numpy as np
from scipy.special import ndtri

DAY = 86400
ArrayLike = Union[int, float, np.ndarray]


@dataclass(frozen=True)
class LabelPolicy:
    delay_median_days: float = 35.0     # median chargeback delay of a fraud
    delay_sigma: float = 0.6            # lognormal shape (0 => deterministic delay)
    delay_min_days: float = 1.0
    delay_cap_days: float = 120.0
    maturity_days: float = 90.0         # no fraud label by then => treated as legit
    train_gap_days: float = 7.0         # rows this close to the cutoff are not trained on
    seed: int = 20240601                # changes the per-id draw, never within one system

    def __post_init__(self):
        if not (0 < self.delay_min_days <= self.delay_cap_days):
            raise ValueError("need 0 < delay_min_days <= delay_cap_days")
        if self.maturity_days <= 0:
            raise ValueError("maturity_days must be > 0")
        if self.delay_sigma < 0:
            raise ValueError("delay_sigma must be >= 0")

    @classmethod
    def from_cfg(cls, cfg: Optional[Mapping]) -> "LabelPolicy":
        cfg = dict(cfg or {})
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in names and v is not None})

    @classmethod
    def fixed(cls, days: float, **kw) -> "LabelPolicy":
        """Legacy rule: every label (fraud and legit) is known exactly ``days`` later."""
        return cls(delay_median_days=days, delay_sigma=0.0, delay_min_days=days,
                   delay_cap_days=days, maturity_days=days, **kw)

    def replace(self, **kw) -> "LabelPolicy":
        return LabelPolicy(**{**asdict(self), **kw})

    def to_dict(self) -> Dict:
        return asdict(self)


DEFAULT_POLICY = LabelPolicy()


# ─────────────────────────────────────────────────────────────────────────────
# deterministic per-transaction randomness
# ─────────────────────────────────────────────────────────────────────────────

def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        z = x
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _uniform_from_id(txn_id: np.ndarray, seed: int) -> np.ndarray:
    ids = np.asarray(txn_id).astype(np.int64).astype(np.uint64)
    salt = _splitmix64(np.array([seed], dtype=np.int64).astype(np.uint64))
    with np.errstate(over="ignore"):
        mixed = ids * np.uint64(0xD1B54A32D192ED03) + salt
    z = _splitmix64(mixed)
    return ((z >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)       # strictly in (0, 1)


def _ret(x: np.ndarray, scalar: bool):
    return x.reshape(()).item() if scalar else x


def fraud_delay_sec(txn_id: ArrayLike, policy: LabelPolicy = DEFAULT_POLICY) -> ArrayLike:
    """Seconds between a transaction and the moment its fraud label becomes known."""
    scalar = np.ndim(txn_id) == 0
    ids = np.atleast_1d(np.asarray(txn_id))
    if policy.delay_sigma == 0:
        d = np.full(ids.shape, policy.delay_median_days, dtype=np.float64)
    else:
        z = ndtri(_uniform_from_id(ids, policy.seed))
        d = np.exp(math.log(policy.delay_median_days) + policy.delay_sigma * z)
    d = np.clip(d, policy.delay_min_days, policy.delay_cap_days)
    return _ret(np.maximum(np.rint(d * DAY), 1).astype(np.int64), scalar)


def fraud_known_at(txn_id: ArrayLike, t: ArrayLike, policy: LabelPolicy = DEFAULT_POLICY) -> ArrayLike:
    """Absolute time at which the fraud label of this transaction would be known."""
    scalar = np.ndim(txn_id) == 0 and np.ndim(t) == 0
    out = np.asarray(t, dtype=np.int64) + np.asarray(fraud_delay_sec(txn_id, policy))
    return _ret(np.asarray(out), scalar)


def fraud_known_by(txn_id: ArrayLike, t: ArrayLike, now: ArrayLike, policy: LabelPolicy = DEFAULT_POLICY):
    return np.asarray(fraud_known_at(txn_id, t, policy)) <= np.asarray(now)


def maturity_at(t: ArrayLike, policy: LabelPolicy = DEFAULT_POLICY):
    return np.asarray(t, dtype=np.int64) + int(round(policy.maturity_days * DAY))


def is_mature(t: ArrayLike, now: ArrayLike, policy: LabelPolicy = DEFAULT_POLICY):
    return maturity_at(t, policy) <= np.asarray(now)


def known_label(y: ArrayLike, txn_id: ArrayLike, t: ArrayLike, now: ArrayLike,
                policy: LabelPolicy = DEFAULT_POLICY) -> np.ndarray:
    """Label as visible at ``now``: 1.0 known fraud, 0.0 mature w/o fraud report, NaN pending."""
    y = np.asarray(y, dtype=np.float64)
    fraud_known = (y == 1) & np.asarray(fraud_known_by(txn_id, t, now, policy))
    mature = np.asarray(is_mature(t, now, policy))
    return np.where(fraud_known, 1.0, np.where(mature & ~np.isnan(y), 0.0, np.nan))


class LabelEvents(NamedTuple):
    """Absolute times (seconds, +inf = never) at which a row enters each label counter."""
    mature_time: np.ndarray            # row counts as a labelled row            (n)
    fraud_time: np.ndarray             # row counts as a known fraud, any age     (fraud)
    fraud_mature_time: np.ndarray      # known fraud AND mature                   (fraud_mature)


def label_ready_times(y: np.ndarray, txn_id: np.ndarray, t: np.ndarray,
                      policy: LabelPolicy = DEFAULT_POLICY) -> LabelEvents:
    """Event times used by the vectorised feature code. Unlabelled rows (y NaN) never count."""
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.int64)
    lab = ~np.isnan(y)
    mature = np.asarray(maturity_at(t, policy), dtype=np.float64)
    fk = np.asarray(fraud_known_at(txn_id, t, policy), dtype=np.float64)
    fraud = lab & (y == 1)
    inf = np.inf
    return LabelEvents(
        mature_time=np.where(lab, mature, inf),
        fraud_time=np.where(fraud, fk, inf),
        fraud_mature_time=np.where(fraud, np.maximum(fk, mature), inf),
    )


def training_target(y: np.ndarray, txn_id: np.ndarray, t: np.ndarray, cutoff: float,
                    policy: LabelPolicy = DEFAULT_POLICY,
                    gap_days: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Target a model trained at ``cutoff`` would have: (target 0/1, usable mask).

    usable  : t < cutoff - gap (rows in the gap are excluded) and the row is labelled
    target  : 1 only if the fraud label is known by the cutoff; unreported fraud = 0
    """
    gap = policy.train_gap_days if gap_days is None else gap_days
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.int64)
    usable = (t < cutoff - gap * DAY) & ~np.isnan(y)
    known = np.asarray(fraud_known_by(txn_id, t, cutoff, policy))
    target = ((y == 1) & known).astype(np.int8)
    return target, usable


def policy_summary(policy: LabelPolicy) -> str:
    return (f"fraud label delay lognormal(median {policy.delay_median_days:g}d, sigma {policy.delay_sigma:g}, "
            f"cap {policy.delay_cap_days:g}d) | maturity {policy.maturity_days:g}d | "
            f"train gap {policy.train_gap_days:g}d")

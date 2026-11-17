"""
Phase 5 -- production-usable blend + probability calibration.

Rank-averaging is robust for combining differently-scaled models, but a raw rank is
computed *within* a batch, so it cannot score a single live transaction. Here each
model's score is mapped to its PERCENTILE in a stored reference distribution (the
out-of-fold predictions), and the percentiles are blended with the OOF-chosen weights.

Two outputs, deliberately separate:

  * ``score(preds)``        the uncalibrated blend. THIS is what ranks transactions and
                            what AUC-PR / recall@FPR / top-k are computed on. Continuous,
                            so no artificial ties.
  * ``predict_proba(preds)`` a calibrated fraud probability = isotonic(score) plus a tiny
                            strictly-increasing term in the score. Isotonic regression is
                            a step function (26% of validation rows tied at p=0 once), so
                            on its own it destroys the ranking at the top and bottom; the
                            tiebreak (weight TIE_EPS) keeps ``predict_proba`` order-
                            identical to ``score`` while staying calibrated to ~1e-6.
                            Serving uses it for cost-based decisions.

Percentile saturation guard: a model refit on more data can be more confident than every
OOF reference value (percentile would saturate at exactly 1.0 and tie). Beyond the
reference range the percentile keeps growing smoothly and strictly monotonically
(rational extrapolation in units of the reference spread) inside (1 - 0.5/n, 1).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.isotonic import IsotonicRegression

TIE_EPS = 1e-6


class CalibratedRankBlender:
    def __init__(self, weights: Dict[str, float], max_ref: int = 200_000, seed: int = 0):
        self.weights = {k: float(v) for k, v in weights.items() if v > 0}
        self.max_ref = max_ref
        self.seed = seed
        self.refs: Dict[str, np.ndarray] = {}
        self.calibrator: IsotonicRegression | None = None

    def _pct(self, name: str, p: np.ndarray) -> np.ndarray:
        """Percentile of ``p`` in the model's OOF reference; interpolated, smooth tails."""
        ref = self.refs[name]
        n = len(ref)
        p = np.asarray(p, dtype=np.float64)
        lo, hi = 0.5 / n, 1.0 - 0.5 / n
        inside = np.interp(p, ref, (np.arange(n) + 0.5) / n)
        spread = max(float(ref[-1] - ref[0]), 1e-12)
        up = np.clip((p - ref[-1]) / spread, 0.0, None)
        dn = np.clip((ref[0] - p) / spread, 0.0, None)
        out = np.where(p > ref[-1], hi + (1.0 - hi) * up / (1.0 + up), inside)
        out = np.where(p < ref[0], lo - lo * dn / (1.0 + dn), out)
        return out

    def blend(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        tot = sum(self.weights.values())
        return sum(w * self._pct(n, preds[n]) for n, w in self.weights.items()) / tot

    def score(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        """Uncalibrated blend in (0, 1): the ranking score."""
        return self.blend(preds)

    def fit(self, oof: Dict[str, np.ndarray], y: np.ndarray) -> "CalibratedRankBlender":
        rng = np.random.default_rng(self.seed)
        for n in self.weights:
            ref = np.asarray(oof[n], dtype=np.float64)
            if len(ref) > self.max_ref:
                ref = ref[rng.choice(len(ref), self.max_ref, replace=False)]
            self.refs[n] = np.sort(ref)
        b = self.blend(oof)
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(b, y)
        return self

    def calibrate(self, score: np.ndarray) -> np.ndarray:
        """Calibrated probability from a ranking score; strictly increasing in the score."""
        score = np.asarray(score, dtype=np.float64)
        iso = self.calibrator.predict(np.clip(score, 0.0, 1.0))
        return (1.0 - TIE_EPS) * iso + TIE_EPS * np.clip(score, 0.0, 1.0)

    def predict_proba(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        return self.calibrate(self.blend(preds))

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "CalibratedRankBlender":
        with open(path, "rb") as f:
            return pickle.load(f)


def calibration_report(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Dict:
    """Brier score, expected calibration error and a reliability table (quantile bins)."""
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows, ece = [], 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        pred, obs = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(pred - obs)
        rows.append({"bin": b, "n": int(m.sum()), "mean_pred": round(pred, 4), "observed": round(obs, 4)})
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece), "reliability": rows}

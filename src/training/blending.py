"""
Phase 5 — production-usable blend + probability calibration.

Rank-averaging is robust for combining differently-scaled models, but a raw rank is
computed *within* a batch, so it cannot score a single live transaction. Here each
model's score is mapped to its PERCENTILE in a stored reference distribution (the
out-of-fold predictions), the percentiles are blended with the OOF-chosen weights,
and an isotonic calibrator (also fitted on OOF rows, i.e. time-respecting, never on
the val holdout) turns the blend into a real fraud probability. Cost-based decisions
(p × amount × loss rate, Phase 7) need calibrated probabilities; AUC-PR doesn't change.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.isotonic import IsotonicRegression


class CalibratedRankBlender:
    def __init__(self, weights: Dict[str, float], max_ref: int = 200_000, seed: int = 0):
        self.weights = {k: float(v) for k, v in weights.items() if v > 0}
        self.max_ref = max_ref
        self.seed = seed
        self.refs: Dict[str, np.ndarray] = {}
        self.calibrator: IsotonicRegression | None = None

    def _pct(self, name: str, p: np.ndarray) -> np.ndarray:
        ref = self.refs[name]
        return np.searchsorted(ref, np.asarray(p, dtype=np.float64), side="right") / len(ref)

    def blend(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        tot = sum(self.weights.values())
        return sum(w * self._pct(n, preds[n]) for n, w in self.weights.items()) / tot

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

    def predict_proba(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        return self.calibrator.predict(self.blend(preds))

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

"""ModelPolicy: a DecisionPolicy that learns ONLY from what the loop shows it.

Inputs it ever sees: the public pre-decision frame (+ causal hist_* features), the previous epochs' decisions /
approvals / logged propensities (``batch.history``) and the labels delivered through ``observe``.
It never touches ground-truth columns: the engine does not hand those over and
tests/test_closed_loop.py checks that flipping them changes nothing.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd

from src.simulator.loop import DecisionPolicy, Decisions, EpochBatch, RulesPolicy

from .actions import CostRule, SegmentThresholdRule, ThresholdRule
from .config import COST_DEFAULTS, MODEL, POLICY
from .features import Featurizer
from .store import OutcomeStore
from .triggers import CalendarTrigger


class ModelPolicy(DecisionPolicy):
    def __init__(self, name: str, rule, trigger=None, *, warmup_days: int = POLICY["warmup_days"],
                 gap_days: int = POLICY["gap_days"], neg_keep: float = POLICY["neg_keep"], train_on_all: bool = False,
                 use_ipw: bool = True, max_weight: float = POLICY["max_weight"], model_params: Dict | None = None,
                 seed: int = 0, cost: Dict | None = None, tune: str | None = None, bootstrap: DecisionPolicy | None = None,
                 tune_days: int = 21, min_pos: int = 15):
        self.name, self.rule = name, rule
        self.trigger = trigger or CalendarTrigger(14)
        self.warmup_days, self.gap_days, self.neg_keep = warmup_days, gap_days, neg_keep
        self.train_on_all, self.use_ipw, self.max_weight = train_on_all, use_ipw, max_weight
        self.params = dict(MODEL, **(model_params or {}))
        self.seed, self.cost, self.tune, self.tune_days, self.min_pos = seed, dict(COST_DEFAULTS, **(cost or {})), tune, tune_days, min_pos
        self.bootstrap = bootstrap or RulesPolicy()
        self.fz, self.store = Featurizer(), OutcomeStore()
        self.model = None
        self.last_train_day = -10 ** 6
        self.snapshots: List[Dict] = []
        self.retrains: List[Dict] = []
        if hasattr(self.trigger, "p"):
            self.trigger.p["gap_days"] = gap_days

    # ── learning ──
    def observe(self, labels: pd.DataFrame, day: int) -> None:
        self.store.observe(labels)

    def _fit(self, X, y, w, seed):
        import lightgbm as lgb
        rng = np.random.default_rng(seed)
        keep = (y == 1) | (rng.random(len(y)) < self.neg_keep)
        w2 = np.where(y == 1, w, w / self.neg_keep)[keep]
        m = lgb.LGBMClassifier(random_state=seed, **self.params)
        m.fit(X[keep], y[keep], sample_weight=w2)
        return m

    def _train(self, day: int, reason: str) -> bool:
        ts = self.store.training_set(day, self.gap_days, self.train_on_all, use_ipw=self.use_ipw, max_weight=self.max_weight)
        if ts is None:
            return False
        X, y, w, amt, seg, _ = ts
        if len(y) < 1000 or y.sum() < self.min_pos:
            return False
        t0 = time.time()
        seed = self.seed + day
        model = self._fit(X, y, w, seed)
        if self.tune and hasattr(self.rule, "fit"):
            # thresholds are tuned on the most recent rows with a model that did not see them
            cut = day - self.gap_days - self.tune_days
            tr = self.store.training_set(day, self.gap_days, self.train_on_all, last_day=cut, use_ipw=self.use_ipw, max_weight=self.max_weight)
            va = self.store.training_set(day, self.gap_days, self.train_on_all, first_day=cut + 1, use_ipw=self.use_ipw, max_weight=self.max_weight)
            if tr is not None and va is not None and tr[1].sum() >= self.min_pos and va[1].sum() >= 8:
                vm = self._fit(tr[0], tr[1], tr[2], seed + 1)
                pv = vm.predict_proba(va[0])[:, 1]
                self.rule.fit(pv, va[1], va[2], va[3], va[4], self.cost, segmented=(self.tune == "segment"))
        self.model = model
        self.last_train_day = day
        self.store.cols = self.fz.columns
        info = dict(day=day, reason=reason, n_train=int(len(y)), n_pos=int(y.sum()), w_pos=float(w[y == 1].sum()),
                    sec=round(time.time() - t0, 2))
        if hasattr(self.rule, "thresholds"):
            info["thresholds"] = dict(self.rule.thresholds, _global=self.rule.default)
        self.retrains.append(info)
        self.snapshots.append(dict(day=day, model=model, rule=_clone_rule(self.rule)))
        return True

    # ── deciding ──
    def decide(self, batch: EpochBatch) -> Decisions:
        day, f = batch.day, batch.frame
        self.store.ingest_outcomes(batch.history)
        X = self.fz.transform(f)
        self.store.cols = self.fz.columns
        if day >= self.warmup_days:
            if self.model is None:
                self._train(day, "initial")
            else:
                fire, why = self.trigger.should_retrain(day, self.last_train_day, self.store)
                if fire:
                    self._train(day, why or "trigger")
        if self.model is None:
            dec = self.bootstrap.decide(batch)
            p = np.asarray(dec.score, float)
            self.store.add_batch(day, f, X, p)
            return dec
        p = self.model.predict_proba(X)[:, 1]
        action = self.rule.decide(f, p)
        self.store.add_batch(day, f, X, p)
        return Decisions(np.asarray(action, dtype=object), p)


def _clone_rule(rule):
    import copy
    return copy.deepcopy(rule)

"""Configuration for the closed-loop experiments (world sizes, model, drift thresholds, cost parameters)."""
from __future__ import annotations

import copy
from typing import Dict

from src.simulator.config import SimConfig, _cb_median, _report_delay

# ───────────────────────── cost model (USD) ─────────────────────────
# Every number is a documented assumption, not a calibrated truth. They are used twice: by the
# expected-cost action rule (using p(fraud)) and by the OUTCOME accounting (using what really happened),
# so a policy is judged by the same prices it optimises.
COST_DEFAULTS: Dict[str, float] = {
    "fraud_loss_frac": 1.0,          # share of the amount lost on an approved fraud (goods + fees are not recovered)
    "chargeback_fee": 15.0,          # fixed fee per approved fraud that becomes a chargeback
    "margin_rate": 0.15,             # contribution margin lost when a legit sale is not completed
    "churn_cost": 12.0,              # churn proxy: P(customer leaves after a false decline) x lifetime value (~0.12 x $100)
    "stepup_fee": 0.10,              # 3DS authentication fee / friction handling per step-up
    "stepup_legit_pass": 0.93,       # legit customers who complete a 3DS challenge (the engine uses the same default)
    "stepup_fraud_pass": 0.20,       # stolen-credential fraud that passes 3DS (engine: 0.05-0.35 by credential type)
    "stepup_churn_frac": 0.0,        # an abandoned 3DS challenge costs the margin but (default) no churn
    "liability_shift_share": 0.9,    # share of a fraud loss that moves to the issuer when 3DS was passed
    "review_cost": 3.5,              # analyst minutes per case (~$3.5)
    "review_accuracy": 0.90,         # analyst catches this share of fraud (engine default)
    "review_false_decline": 0.03,    # legit rows an analyst wrongly declines (engine uses 0.03)
    "review_abandon": 0.01,          # legit customers lost to the review delay
    "review_capacity_frac": 0.004,   # analysts review at most this share of the day's rows
}

# ───────────────────────── world sizes ─────────────────────────
# Delay-compressed world: every label / report delay is halved (and the label window shortened) so a
# 160-day world has the label maturity dynamics of a ~320-day one. Documented in docs/CLOSED_LOOP.md.
DELAY_SCALE = 0.5


def make_world_cfg(seed: int, size: str = "default", **over) -> SimConfig:
    sizes = {
        "quick":   dict(n_customers=1200, n_merchants=200, n_days=100, val_days=5, test_days=20, n_gangs=6),
        "default": dict(n_customers=8000, n_merchants=900, n_days=160, val_days=10, test_days=30, n_gangs=14),
        "full":    dict(n_customers=20000, n_merchants=2000, n_days=300, val_days=20, test_days=50, n_gangs=30),
    }
    base = dict(sizes[size])
    base.update(seed=seed, target_fraud_rate=0.01, adaptive_attackers=True, explore_rate=0.0,
                novel_share_of_test_fraud=0.30)   # more bust_out incidents so detection stats are not n=1
    base.update(delay_compression())
    base.update(over)
    return SimConfig.from_dict(base)


def delay_compression(scale: float = DELAY_SCALE) -> dict:
    cb = _cb_median()
    return dict(cb_delay_median_days=38.0 * scale, label_window_days=int(120 * scale),
                cb_median_by_scenario={k: v * scale for k, v in cb.items()},
                report_delay_days={k: v * scale for k, v in _report_delay().items()})


MODEL = dict(n_estimators=120, learning_rate=0.08, num_leaves=31, min_child_samples=30, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, max_bin=127, n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)
MODEL_QUICK = dict(MODEL, n_estimators=40)

# policy timeline (sim days)
POLICY = dict(warmup_days=50, gap_days=30, retrain_every=14, neg_keep=0.3, decline_threshold=0.10,
              max_weight=100.0, review_threshold=None)
POLICY_QUICK = dict(POLICY, warmup_days=35, gap_days=20, retrain_every=14)

DRIFT = dict(check_every=7, window_days=7, ref_days=14, score_psi=0.10, feat_psi=0.20, feat_psi_k=3,
             perf_drop=0.35, perf_min_pos=12, min_gap_days=14)


def copy_cfg(d: dict) -> dict:
    return copy.deepcopy(d)

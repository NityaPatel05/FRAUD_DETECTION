# Closed-loop experiments (Phase C2)

Four experiments on the v2 simulator, each with a **learning policy in the loop**: it retrains periodically and uses
**only what a real system would have** (the pre-decision frame, the engine's causal `hist_*` features, previous
decisions / approvals / logged propensities, and the labels the loop *delivered* through `observe`).
`is_fraud_true` is used only to score outcomes. See `docs/REAL_SYSTEM_GAP_ANALYSIS.md` (Phase C) for why these matter
and `docs/SYNTHETIC_DATA.md` for the simulator.

Run: `venv\Scripts\python.exe scripts\run_closed_loop.py` (about 25 min on the shared CPU box, 3 seeds).
Smoke run: `--quick` (about 2 min; the world is too small for the numbers to mean anything).
Output: `artifacts/v2/closed_loop/metrics.json`. Tests: `venv\Scripts\python.exe -m pytest -q tests/test_closed_loop.py` (16 tests, about 16 s).

## Method

**World** (`src/closed_loop/config.py`): 8,000 customers, 160 days, 1% fraud attempts (about 175k rows), adaptive attackers,
realistic label gates. All label/report delays are **halved** (chargeback median 19 d, window 60 d) so a 160-day
world has the label-maturity dynamics of a ~320-day one. bust_out exists only in the last 30 days (raised to 30% of
test-period fraud so there are about 7 incidents per world instead of 1). Prevalence is not calibrated to the real world.

**Learning policy** (`policy.py`, `store.py`): days 0-49 are decided by the engine's default rules (bootstrap). From day 50 a
LightGBM (120 trees) is trained on rows older than a 30-day maturity gap whose outcome is observable (approved by
the engine, or analyst-reviewed). A row's target is 1 if a positive label has been delivered so far, else 0 (pending
frauds are negatives, as in production). Declined rows have no label and are excluded. Rows with logged propensity
p (explored would-be declines) get weight 1/p (`weights.py`). Negatives are subsampled 30% with compensating weights.
Features (`features.py`) are the public columns plus the engine's causal `hist_*` aggregates, not the heavier
`sim_track` set, which needs a whole-frame pass and cannot run inside the daily loop. The model is therefore weaker
than the offline sim track (AUC-PR around 0.35-0.5 on the evaluation stream).

**Fixed held-out evaluation stream**: one separate world (seed 777, rules policy, 175k rows, 1,897 frauds) recorded with its
causal features. Every model snapshot is scored on its days 50-129 (bust_out excluded) for AUC-PR, recall@1%FPR and VDR@1%FPR
using true fraud labels. "Deployed over time" scores each 14-day period with the model that was live at its start.

**Outcome accounting** (`outcomes.py`, truth used here only): fraud loss = approved true fraud x (amount + $15 chargeback fee) x
(1 - liability shift x 0.9); false-decline cost = margin 15% x amount + $12 churn proxy; plus review cost, 3DS fee and
3DS-abandonment cost. All defaults are in `COST_DEFAULTS` (`config.py`), each documented there.
Exploration cost = fraud loss on explored rows minus margin earned on explored legit rows.

**Pairing**: "no exploration" runs with explore_rate=1e-9 rather than 0, so the engine consumes the same random numbers in
every arm and arms with the same seed share a world until their decisions first differ.

All numbers: mean of 3 seeds (1, 2, 3), days 50-129 (the pre-test period after warm-up) unless stated; per-seed values are in
`metrics.json` (`per_seed`, and `summary[arm].per_seed`). **Seed-to-seed spread is large** (fraud loss varies by 1.5-2x between
seeds of the same arm), so differences below roughly 20-30% in $ metrics, or 0.03 in AUC-PR, are not distinguishable from noise.

## Experiment 1: selection bias vs exploration

Decline threshold p >= 0.10, retrain every 14 days (8 retrains).

| arm | final-model AUC-PR | final R@1%FPR | deployed-over-time AUC-PR | fraud loss $ | false declines | false-decline rate | exploration rows | exploration net cost $ |
|---|---|---|---|---|---|---|---|---|
| no exploration | 0.340 | 0.450 | 0.260 | 38,338 | 183 | 0.21% | 0 | 0 |
| eps 2% + IPW | 0.355 | 0.464 | 0.271 | 56,606 | 197 | 0.22% | 8 | 882 |
| eps 5% + IPW | 0.377 | 0.484 | 0.267 | 62,702 | 191 | 0.22% | 19 | 346 |
| eps 2%, no IPW | 0.371 | 0.469 | 0.269 | 58,871 | 188 | 0.21% | 6 | -7 |
| oracle labels (upper bound) | 0.502 | 0.570 | 0.364 | 47,862 | 677 | 0.76% | n/a | n/a |

Per-seed fraud loss (seeds 1/2/3): no exploration 51k/28k/36k; eps 2% 67k/65k/38k; eps 5% 70k/62k/55k; oracle 47k/27k/70k.

What showed:
- **Selection bias is real and large.** The oracle-label control beats the biased model by about 0.16 AUC-PR and 12 points of
  recall@1%FPR on the same held-out stream. The gap exists from the start: at day 50 (trained right after the bootstrap rules declined the obvious fraud
  without labelling it) the oracle model has AUC-PR 0.40 vs 0.16 for the biased one (3-seed mean).
- **Exploration recovered only a small part of that gap**: +0.015 (2%) and +0.037 (5%) AUC-PR, +1.4 and +3.4 points of
  recall, positive in most seeds but inside the noise band. It does not approach the oracle.

What did not show:
- **IPW did not help**: eps 2% without IPW is as good as with it (0.371 vs 0.355). With only 6-19 explored rows in the whole
  run (the policy declines about 0.4% of traffic, so 2% of that is a handful of rows), weights of 20-50 on a handful of rows
  mostly add variance. IPW needs far more declined volume or a higher eps to pay off.
- **Live fraud loss did not fall with exploration; it rose** ($57-63k vs $38k). The direct cost of exploration is tiny
  ($0.3-0.9k), so this is not an exploration cost; it is world divergence under adaptive attackers (a few large incidents
  dominate loss) plus seed noise. We cannot claim exploration reduces loss here.
- The oracle arm has *higher* loss than the biased arm in 1 of 3 seeds and 4x the false declines (it learns to flag more).
  "Best offline quality" and "lowest live loss" are different things.
- Caveat: the oracle arm uses `label_mode="oracle"`, which also changes world dynamics (faster card blocking), so it is an upper
  bound on label quality, not a perfectly paired comparison.

## Experiment 2: retraining strategy under adaptive attackers

All arms: threshold p >= 0.10, eps 2% + IPW. "Final model" = what each seed ended with, scored on the fixed stream.

| arm | retrains (mean) | final AUC-PR | final R@1%FPR | deployed-over-time R@1%FPR | fraud loss $ | test-period loss $ (30 d) |
|---|---|---|---|---|---|---|
| no retrain | 1.0 | 0.151 | 0.273 | 0.256 | 44,343 | 87,003 |
| calendar 30 d | 4.0 | 0.349 | 0.432 | 0.338 | 60,489 | 108,405 |
| calendar 14 d (= exp 1 eps 2%) | 8.0 | 0.355 | 0.464 | 0.371 | 56,606 | 81,973 |
| drift-triggered | 1.7 | 0.225 | 0.350 | 0.271 | 43,950 | 75,083 |

Drift trigger: score PSI and feature PSI (the `src/novelty/drift.py` PSI; 16 monitored features) plus AUC-PR on newly matured
labels vs the first matured window; check weekly after a 21-day reference/current window.

What showed: **staleness is real**: a never-retrained model ends at AUC-PR 0.15 vs 0.35 for retrained ones, and its recall on the
fixed stream decays over time (aligned series in `metrics.json`). Retraining more often is monotonically better on the
held-out stream (14 d > 30 d > drift > never).

What did not show:
- **The drift trigger did not track the drift.** Score PSI stayed at 0.002-0.017 (threshold 0.10) and no feature reached PSI 0.2
  at any check: attacker adaptation moves under 1% of rows, so PSI on the whole traffic is blind to it. The only trigger that
  ever fired (twice across 3 seeds) was the delayed-label performance check. That monitor is itself biased (it only sees
  approved rows, whose high-score fraud was declined), and it needs labels that are 30+ days old. Result: 1.7 retrains vs 4 for
  the calendar arm, recall between "never" and "calendar 30 d". In this simulator a calendar schedule is the better policy;
  drift triggers would need segment-level or tail-of-score signals to be useful (not tried here).
- **Live fraud loss does not rank the arms the way held-out quality does.** No-retrain and drift arms have the *lowest* live loss
  ($44k) and calendar 30 d the highest ($60k) despite the worst/best models. Per seed the arms differ by tens of thousands in
  both directions; adaptive attackers and a few large incidents dominate. Held-out AUC-PR/recall are the reliable signal here;
  live loss with 3 seeds is not.

## Experiment 3: cost-based decisions

Retrain every 14 d, eps 2% + IPW. "Single threshold" and "per-segment thresholds" are tuned on a held-out recent slice by
minimising fraud loss + false-decline cost (global threshold vs channel x new/established segments); "expected cost" picks
argmin of expected cost over approve / step_up_3ds / review / decline from p(fraud) and the amount
(`actions.py::expected_costs`), review limited to 0.4% of rows per day (largest saving first).

| arm | net cost $ | fraud loss $ | false declines | false-decline cost $ | 3DS-abandon cost $ | 3DS rate | reviews (load) | review cost $ |
|---|---|---|---|---|---|---|---|---|
| single threshold | 67,605 | 65,677 | 68 | 1,928 | 0 | 0 | 0 | 0 |
| per-segment thresholds | 46,732 | 42,883 | 116 | 3,849 | 0 | 0 | 0 | 0 |
| expected-cost actions | 52,665 | 47,375 | 4 | 1,555 | 2,200 | 2.5% | 375 (0.35%) | 1,314 |

Per-seed net cost (1/2/3): single 57k/66k/80k; segment 52k/46k/43k; expected-cost 43k/52k/63k.

What showed: both richer rules beat the single threshold in **all 3 seeds** on net cost (-31% and -22% on average), driven by
lower fraud loss. Expected-cost selection nearly eliminates false declines (4 vs 68-116) by using 3DS and review instead
of declining, at a cost of 2.5% of traffic stepped up and 0.35% reviewed.

What did not show: expected-cost did **not** beat the tuned per-segment thresholds (better in 1 seed of 3, worse in 2). Its
fraud loss is higher: p is trained on selection-biased labels and not recalibrated, so the cost rule over-trusts low scores
(its break-even p of about 0.2 is rarely reached) and approves or steps up risky rows. The single-threshold arm's high loss
is partly tuning noise (thresholds tuned on a thin slice of positives). Net-cost ranking should be read as "richer actions
help", not as a clean ordering of the two.

## Experiment 4: novelty on the held-out scenario (bust_out)

bust_out is generated only in the last 30 days: established customers on their own device/IP with owner credentials
spend a burst (8-30 rows over 1-4 days) across high-value merchants. Novelty layer = daily segment-burst alerts from
`src/novelty/attack_detection.py::detect_attacks` over mcc, amount band, channel, merchant country, IP type (robust z >= 6, 28-day
baseline, min volume 5); compared with the supervised model's recall at a 1% FPR threshold on the fixed stream's test
period (same model the arm ended with, which never saw bust_out).

| arm (3 seeds pooled) | incidents | flagged by segment alerts | detection delay (days, when flagged) | false alarms pre-test (per week) | supervised bust_out recall@1%FPR | supervised recall, other fraud | live: bust_out rows approved | live bust_out loss $ (30 d) |
|---|---|---|---|---|---|---|---|---|
| eps 2% (calendar 14 d) | 23 | 3 (13%) | 2, 0, 3 | 3.4 | 0.162 | 0.354 | 86% | 46,993 |
| no retrain | 17 | 1 (6%) | 1 | 3.5 | 0.077 | 0.224 | 93% | 44,465 |
| calendar 30 d | 22 | 2 (9%) | 0, 1 | 3.5 | 0.124 | 0.358 | 92% | 53,653 |
| drift-triggered | 21 | 2 (10%) | 0, 1 | 3.5 | 0.067 | 0.244 | 91% | 39,645 |

What did not show: **the novelty layer does not catch bust_out here.** It flags about 10% of incidents, 0-3 days after the incident's first row
(bursts last 1-4 days), while raising about 3.5 false alarms per week in the pre-test period and about 75% of its test-period alerts
cover no bust_out row. This is by construction: bust_out rows sit inside normal segments, and a handful of rows per incident cannot
move a segment count at this volume. Mined rules were not attempted (nothing to mine from a segment that did not alarm).
The supervised model, never trained on bust_out, catches 7-16% of its rows at 1% FPR (vs 22-36% of other fraud), so
it generalises a little (amount, velocity, new merchants) but is also weak. Roughly 86-93% of bust_out rows were approved
live and cost $40-54k per world in 30 days, the largest single loss item in the test period. A real system would need
a customer-level behaviour-change feature (spend vs own baseline) or a credit-exposure rule; segment bursts and the generic model are not
enough. (A planned score-PSI drift series for this experiment came out empty because the last model snapshot was too recent;
it is not reported.)

## What a real system should take from this

- Selection bias costs more than anything else measured here (0.16 AUC-PR). Exploration at 2-5% helps only a little at this decline
  volume, and weights from a handful of explored rows are mostly noise; budget exploration by *number of explored rows
  per retrain window* (hundreds), not by percentage, and keep a labelled holdout slice as the clean measurement instrument.
- Retrain on a schedule; do not rely on feature or score PSI alone to trigger retraining when the shift is a small share of traffic.
  Monitor segments and the tail of the score, and make label-based monitors selection-aware.
- Evaluate on a fixed stream with true outcomes; live fraud loss from a few seeds is too noisy to rank policies.
- Move from a threshold to cost-based actions: the biggest gain is replacing declines by 3DS / review where it is cheaper. Recalibrate p
  on exploration (IPW) data before trusting expected-cost arithmetic.
- Bust-out style first-party fraud needs behaviour-change and exposure features, not segment bursts.

Limits: 3 seeds; one small delay-compressed world; costs are documented assumptions; the model uses the engine's `hist_*` features, not the
full sim-track set; analyst review capacity and accuracy are the engine's; thresholds were not tuned per arm (except in exp. 3).

## Full run (GPU / large machine)

```
venv\Scripts\python.exe scripts\run_closed_loop.py --size full --seeds 1 2 3 4 5
```
`full` = 20,000 customers, 300 days (about 4x the rows); the models are LightGBM, so the GPU only helps the larger tree counts; set
`MODEL["n_estimators"]` and `device` in `src/closed_loop/config.py` if desired. `--exps 1,3` runs a subset; `--reaggregate` rebuilds the
summary blocks of an existing `metrics.json`.

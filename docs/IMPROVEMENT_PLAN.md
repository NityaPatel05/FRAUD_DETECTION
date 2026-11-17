# Fraud Detection — Improvement Plan

Branch: `improve-aucpr`. The plan combines the code audit (4 agents) with research on what production fraud teams do: Stripe, Adyen, Feedzai, Featurespace, PayPal, Booking.com, Meta, Uber, Nubank, Revolut, Monzo and the Fraud Detection Handbook.

## Status (updated)

| Phase | Code | Measured | Where |
|---|---|---|---|
| 0 v2 rebuild | ✅ | ⏳ | `pipeline_v2.py`, `src/training/` |
| 1 ablation harness | ✅ | ⏳ | `scripts/run_ablation.py` → `docs/ABLATIONS.md` |
| 2 data cleanup, derived signals, `--schema raw` | ✅ | ⏳ | `derived_signals.py` |
| 3 behaviour (replaces BiLSTM) incl. HMM | ✅ | ⏳ | `add_behaviour_v3`, `hmm_features.py` (vectorised Baum-Welch) |
| 4 network (replaces GNN) | ✅ | ⏳ | `add_network_v4`, `streaming_ops.py` |
| 5 tuning + calibrated blend | ✅ | ⏳ | `scripts/run_tuning.py`, `src/training/blending.py`; `run_training.py --use-tuned` |
| 6 novelty (replaces VAE) | ✅ | ⏳ | `src/novelty/`, `run_novelty_monitoring.py`, `run_heldout_pattern_test.py` |
| 7 serving: online features (parity-tested), decisions, reasons, API | ✅ | ⏳ | `src/serving/`, `run_policy_simulation.py`, `benchmark_latency.py`, `uvicorn src.serving.api_v2:app` |
| 8 feedback: labels w/ maturity, shadow + promotion gate, monitoring, exploration | ✅ | ⏳ | `src/feedback/`, `src/jobs/retrain_v2.py`, `run_promotion_gate.py`, `run_monitoring_report.py`, `run_exploration_simulation.py` |
| 9 GRU sequence score experiment | ✅ | ⏳ | `src/experiments/`, `run_sequence_experiment.py` (GNN embeddings: deliberately skipped, see Phase 4) |
| 9 synthetic simulator | 🔄 | — | other Claude session (`src/simulator/`, `docs/SYNTHETIC_DATA.md`) |
| GPU training | ✅ | — | `--device gpu` / `training_v2.device` |

Tests (`tests/`, `pytest.ini` limits collection to `tests/`): `test_v2_pipeline.py`, `test_novelty.py`, `test_serving.py`, `test_feedback.py`, `test_experiments.py`, plus A1: `test_label_policy.py`, `test_training_eval.py`, `test_run_guard.py`. They include:
- the no-future-leakage test for every feature;
- an online/offline parity test over all features, with a negative control that proves the check can fail.

### Phase A1 -- "fix the offline system" (2026-11-17, from `docs/REAL_SYSTEM_GAP_ANALYSIS.md`)

The first GPU run proved the pipeline runs but not that the numbers mean anything. Fixed in code (not yet re-run on GPU):

| Problem found | Fix |
|---|---|
| Runner marked steps FAILED after "Report saved" (0xC0000005 at shutdown on Python 3.14); ~11 GPU-hours lost | `src/utils/run_guard.py`: success sentinel + `os._exit`; runner trusts the sentinel, OOM only if the log shows it, lock file, Python 3.11-3.13 only. CatBoost off by default; `--final` reuses the validation run's CV |
| Isotonic step output used for ranking (26% ties at p=0, blend AP < LightGBM) | `CalibratedRankBlender.score()` ranks, `predict_proba` = isotonic + strictly monotone tiebreak; smooth percentile extrapolation |
| Every label assumed known exactly 30 d later; config said 30 d and 90 d | `src/labels/policy.py` + `label_policy` config: per-transaction chargeback delay (lognormal, median 35 d, cap 120), unreported = legit after 90 d. Used by features, training targets (known at cutoff, `train_gap_days`), evaluation. Online engine must adopt it (phase A2) |
| Early stopping on the rows that were scored; 3 x 15-day folds | rolling 5 x 10-day folds, first half of each val window = early stopping only, second half = score |
| Only AP / recall@FPR | value detection rate, CP@k per day, false-decline rate, per-fold base rate and AP/base; slices: cold-start clients, unbuildable uid, copied positives removed |
| LightGBM GPU `max_bin=63`; CatBoost GPU stops on Logloss | `max_bin` 255 (`gpu_max_bin`); CatBoost metric configurable and documented |
| Ablation noise (+-0.003) vs keep threshold 0.005 | `--seeds` (default 3), paired per-fold deltas, KEEP needs gain > 2 x SE and >= 0.005; `--groups` (preset `overlap`) |
| Label encoder fitted on train+val+test; policy tuned and reported on the same val | encoder fitted on train rows (unseen -> NaN); `run_policy_simulation.py` tunes on the first half of val, reports the second |

Expect LOWER numbers than the first run: that is the honest protocol, not a regression. Compare runs only under the same label policy (`--label-median-days 30/60/90`, `--tag-suffix`).

Status: code done, unit-tested; the CPU smoke on a 20% card sample is the only end-to-end run so far. Not yet run on the full data.

### Full run order (on the GPU machine)
```
# features (CPU, minutes)
python scripts/run_preprocessing.py
python scripts/run_feature_engineering.py                  # full schema
python scripts/run_feature_engineering.py --schema raw     # real-world fields only
python scripts/run_feature_selection.py                    # (+ --schema raw)

# measure what earns its place (Phases 1-4)
python scripts/run_ablation.py --device gpu                # (+ --schema raw)

# Phase 5: tune, then train the calibrated ensemble
python scripts/run_tuning.py --model lgbm --device gpu
python scripts/run_training.py --use-tuned --device gpu    # writes bundle_refit.json, val_predictions.parquet (LightGBM + XGBoost; add --models lgbm xgb catboost for CatBoost)
python scripts/run_training.py --schema raw --device gpu   # the model the live system serves

# Phase 6: novelty
python scripts/run_heldout_pattern_test.py --device gpu
python scripts/run_novelty_monitoring.py

# Phase 7: decisions & serving
python scripts/run_policy_simulation.py                    # money, decision rates, review threshold
python scripts/benchmark_latency.py --n 2000               # p99 target < 50 ms
uvicorn src.serving.api_v2:app --port 8000                 # POST /v2/score, /v2/label

# Phase 8: feedback loop
python scripts/run_exploration_simulation.py               # naive vs IPS vs truth
python scripts/run_monitoring_report.py --labels data/processed/val_raw.parquet
python -m src.jobs.retrain_v2 --dry-run                    # retrain → register challenger
python scripts/run_promotion_gate.py --labels data/processed/val_raw.parquet

# Phase 9 (optional)
python scripts/run_sequence_experiment.py --device gpu
python scripts/run_training.py --final --device gpu        # train+val → Kaggle test predictions
```

Known issue (pre-existing, not touched): the uncommitted `src/inference/feature_store.py` has every line indented 4 spaces (IndentationError). This breaks only the legacy `server.py`; the committed version is fine.

---

## Principles

1. **Measure before adding.** Every feature family or component goes through the same ablation and is kept only if it passes the keep rule below.
2. **Past-only (causal) by default.** Every feature at time *t* uses only data before *t*. Labels count only after `label_delay_days`. This makes the score honest and servable online.
3. **One code path.** Offline training features and online serving features must come from the same definitions, checked by a parity test.
4. **The val holdout (days 131–182) is touched only at the end of each phase.** All decisions use the time folds inside the train period.
5. **Tree ensemble on engineered features is the core.** Learned sequence and graph models return only as optional feature generators that pass the keep rule.

### Keep rule (applies to every phase)

Run on the 3 walk-forward time folds, same seeds, with and without the change. Keep the change only if:
- mean CV AUC-PR gain is **≥ +0.005**; and
- it is positive in **≥ 2 of 3 folds**; and
- recall at 1% FPR does not drop.

Record every result, including failures, in `docs/ABLATIONS.md`.

---

## Phase 0: Lock in the v2 rebuild (code done, not yet run)

Already implemented and tested (16 tests): single split, no drift filter, all-rows feature build, label encoding, D fixes, client aggregates, velocity, deviation, lags, first-seen flags, delayed label history, graph statistics, time-fold CV, refit + seed bagging, rank blend, report.

- [ ] Review the diff and commit on `improve-aucpr`.
- [ ] Move `src/models/{vae,bilstm,gnn}` to `src/models/legacy/` with a README explaining why, and remove them from `run_training.py` imports (already done). Keep them out of the inference path when Phase 7 lands.

## Phase 1: First honest baseline (first training run)

Goal: replace the old 0.61 with a number you can trust, and build the tool every later phase depends on.

1. **Ablation harness.** Write `scripts/run_ablation.py`: train LightGBM on the time folds with feature families switched on or off (by name prefix or config flag). It prints per-fold AUC-PR, mean ± std, paired deltas vs baseline, and appends to `docs/ABLATIONS.md`. Use `--fast` settings for speed.
2. Run the pipeline:
   ```
   python scripts/run_preprocessing.py
   python scripts/run_feature_engineering.py                      # causal
   python scripts/run_training.py --models lgbm --fast            # quick sanity
   python scripts/run_feature_selection.py
   python scripts/run_training.py                                 # full ensemble
   python scripts/run_feature_engineering.py --agg-mode transductive
   python scripts/run_training.py --agg-mode transductive         # Kaggle-style comparison
   ```
3. **Sanity checks:**
   - Feature count is ~350–450.
   - No feature has train AUC far above its val AUC.
   - The top-20 importances make sense (aggregates, label history, C/D/M, amount).
   - The run time is acceptable.
4. **Ablate each v2 family against the full set:** label history, graph stats, behaviour (velocity/deviation/lags), group aggregates, count encodings. This shows which v2 parts actually carry the gain.

**Deliverable:** `metrics_val.json` for both modes, plus the first ablation table. **Expected:** the causal score should beat 0.61. Estimate 0.65–0.72, not measured.

## Phase 2: Data cleanup and honest enrichment

Based on the read-only data study, which profiled all 434 columns with single-column time-consistency AUCs (two windows), KS shift and unseen-value rates.

**Key facts that shape the plan:**
- Labels are client-level: 96.6% of UIDs with 2+ transactions are all-fraud or all-legit.
- 42% of val rows have a client seen in train, but only 22.6% of Kaggle-test rows do. The model must work for new clients.
- **Days 1–29 are a different regime (likely December/Christmas):**
  - identity coverage ~33% vs ~9% afterwards;
  - fraud rate 2.5% vs 3.7%;
  - `id_38` flips direction.

  Test days 360–395 return to that regime.
- Device strings go stale. 60.6% of test `id_31` browser strings are never seen in train or val (for example, Chrome 63 → 69).

**Remove (config list `drop_cols` in `fe_v2`, applied in `pipeline_v2`):**
1. Near-constant: V107, V305, V108, V110–V114, V117–V122, C3.
2. Over 99% missing with no signal: id_07, id_08, id_21, id_22, id_24, id_25, id_26, id_27.
3. Exact or near duplicates:
   - D9 (= fractional part of D8, i.e. time of day, already covered by `hour`);
   - D2 (= D1 in 92.8% of rows);
   - C10 (Spearman 0.971 with C8).
4. V columns weak in both time-consistency windows: V41, V100, V104, V135, V286, V296–V300, V311, V319–V321. These are removed only if the ablation agrees.
5. Already excluded in v2: TransactionDT and day as inputs, plus week/month cycles. Do **not** reintroduce `week_sin`/`week_cos`/`month_*`/`is_weekend` from the legacy `time_features.py`.

**Replace raw with a stable derivation (fights stale device strings):**
- `id_31` → browser family plus a version-lag flag (computed from past data only);
- `id_30` → OS family;
- `DeviceInfo` → vendor;
- `id_33` → screen area;
- `id_13` → count-encoded.

Keep the raw columns only if the ablation shows they still help.

**Add (honest derivations, ranked):**
1. **Cross-border / foreign-currency flags** (real-world analogue: card country ≠ IP country):
   - `amt_decimals > 2` (11.7% vs 2.5% fraud);
   - `addr2 != 87` (10.2% vs 2.4%);
   - `card3 != 150` (11.3% vs 2.5%);
   - email country TLD (.mx/.de/.jp/…) vs addr2 mismatch.
2. `is_proxy` and proxy type from `id_23`, the only IP-proxy signal (anonymous proxy 13.7% fraud).
3. Verification / seen-before counts:
   - M1–M9 match, mismatch and missing counts (already present), plus M4 category;
   - `id_12/15/16/28/29` Found / NotFound / New as new-account or new-device flags.

   Describe them as "undocumented match flags", **not** AVS/CVV/3DS.
4. Pseudo device fingerprint `DeviceInfo + id_19 + id_20 + id_31 + id_33`, extending the current `device_fp`. Use it in Phase 4 sharing counts (analogue of BAF `device_distinct_emails_8w`).
5. Hand-built email-domain class for the 59 domains: free webmail / ISP / privacy (protonmail 40.8% fraud) / corporate / placeholder. No domain appears on public disposable-email lists, so that lookup adds nothing.
6. Amount patterns: decimal length, round-amount flag.
7. Optional, then ablate: `days_to_christmas`, assuming day 0 ≈ 2017-12-01 (inferred from the volume peaks on days 22–24 and 389). Train has only one December, so treat it as high-risk.

**Time-consistency fix:** because days 1–29 are a different regime, run `run_feature_selection.py` with two windows (first month → last month, and days 31–130 → last month). Drop a feature only if it fails both.

**Not feasible:** BIN tables (card1 is not a BIN), IP/geo enrichment (no IP field), local hour from addr1 (anonymised region).

**Do NOT:**
- fabricate columns (fake IP, geo, merchant or MCC): they are independent of the label, so they carry no signal;
- generate anything conditioned on `isFraud`: that is target leakage;
- add CTGAN/TVAE columns synthesised from IEEE itself: they add no information;
- use SMOTE or oversampling, which hurts AUC-PR and calibration (Elor 2022; van den Goorbergh 2022; Fraud Detection Handbook ch. 6);
- join rows from a simulator: there are no shared keys, so the join is either random (no signal) or label-matched (leakage).

**Real-world fields IEEE-CIS lacks (state this in the write-up):** true customer and merchant IDs, MCC, terminal ID, POS entry mode, decline codes, IP and geolocation, timezone-aware timestamps, KYC / account age, session events, explicit CVC/AVS/3DS results, chargeback date and reason. To demonstrate those components, use a **separate** simulator module (Phase 9): the Fraud Detection Handbook simulator for terminal risk with label delay, or Feedzai BAF for device/email velocity.

## Phase 3: Customer behaviour features (replaces the BiLSTM)

Production practice: Feedzai profiles, Featurespace behavioural analytics, Sift velocity, Coinbase sequence features. All go in `client_features.py`, causal, vectorised. Ablate each sub-step.

| Step | What | Source |
|---|---|---|
| 3a | Multi-entity velocity: add a 10-min window. Apply count/sum/distinct windows to `card1_addr1` and `device_fp` as well as the client. Add ratios `n_1h / n_7d`, `sum_24h / sum_prev_24h`. | Feedzai, Sift |
| 3b | Exponentially decayed aggregates (half-lives 1h/1d/7d/30d): decayed count, mean and std per client. Add short-minus-long "delayed EMA" burst signals. | Feedzai (½ memory, 2–3× faster than windows) |
| 3c | Habit deviation: how typical this hour is for the client (decayed 24-bin histogram), plus `history_len`. Keep NaN (not 0) when there is no history. | Bahnsen (+13% savings), Featurespace |
| 3d | Last-k lags, k = 2, 3: amounts, gaps, product, and flags for device/email/address changed since the last transaction. | Coinbase, Jurgovsky |
| 3e | HMM likelihood features (`hmmlearn`): 5-state HMMs on windows of 3 (amount, time gap), genuine vs fraud histories, fit on a period **before** the training folds. | Lucas et al.: +9–18% AUC-PR over aggregates on real bank data |

## Phase 4: Network / fraud-ring features (replaces the GNN)

Production practice: Booking.com, PayPal, Adyen ShopperDNA, Meta DEC, Grab. Goes in `graph_features.py` and `label_history.py`.

| Step | What | Source |
|---|---|---|
| 4a | Shared-entity counts in windows (1d/7d/30d): `pair_is_new` (first time this device is seen with this client), pair age, entity age, how fast new links are forming. | Adyen, Stripe, Booking, Grab |
| 4b | 2-hop fraud exposure: delayed-label fraud rate of *other* clients sharing this client's device / card1_addr1, excluding the client's own history. `hops_to_fraud` capped at 3. | Booking, Meta DEC, Feedzai GuiltyWalker |
| 4c | Incremental union-find clustering over hard links (uid2, device_fp, id_19/id_20 fingerprint). Hub cap K = 20. Features: cluster size, known fraud in cluster, 24h growth, `merges_components`. | PayPal, Adyen, arXiv 2512.19061 |
| 4d | Ring motif: number of other clients sharing ≥ 2 identifiers with this one. | Neo4j ring rule, IBM GFP |
| — | Skip DeepWalk/node2vec embeddings (no vector for new accounts). Personalized PageRank and Leiden communities only as analyst tools. | — |

Note: IEEE-CIS has weak identifiers (no IP or phone; DeviceInfo missing on 78.5% of rows), so expect modest gains here. These features matter far more on data with real identifiers.

## Phase 5: Model layer refinements

1. **Tuning:** Optuna on the time folds (LightGBM first, ~50 trials, objective = mean fold AUC-PR). Never tune on val.
2. **CatBoost:** confirm it now adds diversity. Keep it in the blend only if the OOF blend weight is > 0 and the blend beats the best single model.
3. **Calibration:** isotonic calibration of the blended score, fit on the last fold's OOF predictions. Phase 7 needs real probabilities.
4. **Client smoothing:** keep only if the causal-smoothed val AUC-PR beats raw (it is already reported).

## Phase 6: Catching new fraud patterns (replaces the VAE)

Production practice: Uber RADAR, Stripe card-testing defence, Feedzai drift monitoring, Monzo held-out-type tests.

1. **Held-out-pattern test.** Measure, not assume, how well the system generalises: train with one ProductCD's fraud removed (or one time slice), then measure recall on it. Run for the v2 model before and after Phase 3–4.
2. **Segment attack detection.** Hourly series per segment (ProductCD, card4/card6, email domain, device OS, amount band). Robust z-score (median/MAD) on volume and amount. Alert on deviation > k sustained for N hours with minimum volume.
3. **Rule mining and rules engine.** On alert, mine over-represented feature=value combinations with FP-Growth (`mlxtend`). Backtest each candidate rule on history (precision, hits, false-positive cost). Store rules as versioned YAML, mode shadow → enforce, with an expiry date.
4. **Drift monitoring.** PSI on the top-30 features and on the score distribution, plus a domain classifier (recent week vs reference). Write a report via `evidently` or plain pandas.
5. **Isolation Forest, optional.** Only for an analyst discovery queue, or as one feature if it passes the keep rule. Never a blocking score.

## Phase 7: Production decision layer (makes the online path match the offline model)

Production practice: Stripe Radar, Adyen, Sift workflows, Nubank orchestrator, Feedzai Railgun, Chronon/Feast.

1. **Online feature parity (largest task).**
   - Write streaming classes that compute every v2 causal feature from Redis state:
     - EMA state;
     - windowed counters (sorted sets);
     - last-k ring buffer;
     - seen-sets for first-seen flags;
     - delayed-label counters;
     - union-find parents.
   - **Parity test:** replay the val period event by event through the online classes and assert equality with the offline parquet features.
   - Fix the legacy inference path to use v2 models from `artifacts/v2/`.
2. **Decision service:** allow/block lists → hard rules (short-circuit) → calibrated model → policy rules on (p, amount, segment) → approve / step-up (3DS/OTP) / review / decline.
   - Cost-based policy: decline if `p × amount × loss_rate > friction_cost`.
   - The review band is sized to analyst capacity (top-K expected loss per day).
3. **Reason codes:** TreeSHAP top-3 contributions mapped to readable codes, stored with every decision.
4. **Decision log:** model version, rules fired, feature snapshot, score, action, reason codes.
5. **Latency budget:** p99 < 50 ms end to end. Feature fetch ≤ 15 ms, model ≤ 5 ms, rules ≤ 2 ms. Measure it.

## Phase 8: Feedback loop and safe deployment

1. **Label table:** `event_time`, `label_time`, `source` (chargeback / analyst / customer). Retrain uses only matured labels. Optionally add a second model trained on fast analyst labels (Dal Pozzolo).
2. **Weekly retrain:** update `src/jobs/weekly_retrain_job.py` to the v2 pipeline.
3. **Champion/challenger:**
   - The challenger scores live Kafka traffic in shadow mode.
   - Promotion requires better matured-label metrics **and** decision rates (block/review/step-up) at current thresholds staying within bounds (Sift SafetyNet).
4. **Monitoring:** PSI, decision rates, review-queue depth, p99 latency, online/offline feature skew, false-positive-rate drift alerts.
5. **Stripe-style exploration (simulation):** let a small, score-dependent fraction of would-be declines through, log P(allow), and report inverse-propensity-weighted precision and recall next to the naive ones.

## Phase 9: Optional / roadmap

Each item must pass the keep rule:
- **Sequence-embedding experiment:** small unidirectional GRU or CoLES encoder over client histories. Out-of-fold embedding or score as a LightGBM feature. Expected 0 to +0.01 AUC-PR.
- **GNN-embedding experiment:** inductive, time-respecting, hub-capped. Compared against the Phase 4 features with an edge-shuffle control.
- **Second demo dataset** for mule rings (PaySim or IBM AMLworld). Real P2P identifiers make graph methods shine there. State clearly that it is synthetic.

---

## Order, effort, and what each phase proves

| Phase | Effort | Needs training runs | Proves |
|---|---|---|---|
| 0 | 0.5 day | no | — |
| 1 | 1–2 days | yes | the real baseline; which v2 parts matter |
| 2 | 1 day | yes (ablation) | data understanding |
| 3 | 3–4 days | yes | behaviour modelling, the BiLSTM replacement |
| 4 | 3–4 days | yes | network modelling, the GNN replacement |
| 5 | 2 days | yes | tuning and calibration discipline |
| 6 | 3–4 days | partly | novel-fraud handling, the VAE replacement |
| 7 | 5–7 days | no | production serving, parity, decisions, explainability |
| 8 | 3–5 days | partly | MLOps: feedback, shadow deployment, monitoring |
| 9 | optional | yes | advanced deep learning, evaluated honestly |

Headline metrics for the write-up: **AUC-PR, recall at 1% FPR, precision in the top 1%, client precision@500, p99 latency, and business cost saved**, not ROC-AUC alone.

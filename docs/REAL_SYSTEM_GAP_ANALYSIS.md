# Real-system gap analysis (2026-11-17)

Built from five parallel studies:

1. Production fraud systems: Stripe, Adyen, Monzo, Revolut, Coinbase, Uber, Visa/Mastercard, Feedzai, Featurespace.
2. Research on fraud methods and evaluation, 2020–2026.
3. An audit of this pipeline's code.
4. A study of the IEEE-CIS data.
5. A comparison of our simulator with PaySim, AMLworld, BAF and the Fraud Detection Handbook simulator.

The question it answers: is our weakness the data, the system, or both? And what does a real fraud system have that ours lacks?

---

## 1. Verdict: both, and they need different fixes

### The data limits how real-world the system can be

IEEE-CIS is fine for honest model work, but it cannot stand in for a real payment system.

- **Copied labels.** After a chargeback, Vesta also labelled the account's later transactions as fraud. As a result:
  - 62.6% of fraud rows come after that client's first fraud.
  - If the previous transaction was fraud, the next one is fraud 87% of the time.
  - Part of what looks like "detection" is really remembering clients who were already caught. A real system would simply block the card at that point.
- **91% of the columns are masked** (V, most id_, C, D).
- **Fields every real system uses are absent:** merchant, MCC, terminal, IP, geolocation, a stable device ID, real timestamps, authorization or decline codes, 3DS, and chargeback dates (so we can't tell when a label would have become known).
- **The client key is a reconstruction** (card1 + addr1 + day − D1). It can't be built for 11% of rows, and those rows hold **38% of all fraud**.

The evaluation design moves the score more than our features do. LightGBM on raw columns:

| Split | AUC-PR |
|---|---|
| Random row split (what most Kaggle notebooks do) | 0.853 |
| Time split | 0.586 |
| Time split with a 30-day label gap | 0.505 |
| **Clients never seen before** (the part that transfers to production) | **0.373** |
| Clients seen before that already had fraud | 0.976 (memorisation) |

Our v2 features raise the time-split score from 0.586 to 0.667, so the feature work is real. But much of the 0.667 comes from the label-history feature and from clients we have seen before. **When the labels are made realistic, the headline will go down.** That drop is the honest number, not a regression.

### The system has real faults, all confirmed in code

| # | Problem | Effect |
|---|---|---|
| S1 | The runner treats a crash during shutdown, after all results are saved, as a failure. There is no lock against two runs at once. | About 11 h of GPU time wasted. Steps 8–17 never ran, including the raw-schema model, the novelty layer, policy, latency and the simulator track. |
| S2 | The blend is scored on the isotonic calibrator's output, a step function. 26% of val rows tie at exactly 0, and there are ties at the top-1% cut. | The blend scores below LightGBM alone. |
| S3 | Label realism. Offline, every row counts as labelled 30 days later. Online, a label exists only if someone posts it. The config says 30 days in one place and 90 in another. | label_history, the only feature family that passed the keep rule, depends on an assumption production can't meet. |
| S4 | In cross-validation, each model stops early on the same fold it is then scored on. | CV and the out-of-fold predictions feeding the blend are optimistic. |
| S5 | LightGBM on GPU forces max_bin = 63, but high-cardinality codes such as card1 (13k values) go in as plain numbers. | Accuracy is left on the table. |
| S6 | The gap between training and validation folds is 7 days, shorter than the label delay. Labels younger than the training cutoff are still used. | Optimistic compared with production. |
| S7 | Ablation noise between identical runs (±0.003) is about the size of the keep threshold (0.005). Removing one family at a time can't detect overlapping families. | "Keep" and "drop" verdicts are unreliable. |
| S8 | The online feature engine isn't safe to deploy: not thread-safe, re-scoring the same transaction double-counts, state grows without limit, labels for unknown transactions are silently dropped, the snapshot isn't checked against the model, and the challenger is scored on the champion's features. The default bundle is full-schema, which needs Vesta-only columns. | The live path isn't production-grade. |
| S9 | Minor: the review threshold is tuned and evaluated on the same val set; categories are label-encoded using train + val + test; CatBoost on GPU stops on Logloss; `--final` repeats the whole CV. | Small biases and wasted time. |

What the audit checked and found clean: no future-information leakage in any feature. Aggregates use prior rows only, label statistics are delayed, and val is never used for tuning or selection.

---

## 2. Real systems compared with ours

Capabilities are listed in the order production evidence ranks them.

| # | What real systems do (source) | Ours | Gap |
|---|---|---|---|
| 1 | **Features computed only from data available at the time, defined once and computed identically online and offline, with a skew monitor** (Stripe Chronon/Shepherd; Coinbase saw model decay from feature skew) | Causal features plus a replay-based parity test | **Partial.** The parity test passes only because every label is posted. Label meaning differs between online and offline (S3, S8). |
| 2 | **Velocity and entity-profile aggregates over several time windows, often shared across merchants** (Stripe: "cards per IP", "countries in 24 h") | Rich per-client aggregates | **OK on IEEE.** No IP, merchant or device ID to build them from, so it can only be shown on the simulator. |
| 3 | **Label pipeline that respects maturity:** several label sources (chargebacks, issuer fraud reports, manual review, customer reports); a legit label only after 90–120 days; chargebacks can take up to 120 days | One label at a fixed 30 days | **Missing.** Biggest realism gap. |
| 4 | **Exploration or holdout traffic with logged propensities:** let about 5% of would-be blocks through and correct with inverse-propensity weights (Stripe, Booking.com) | `run_exploration_simulation.py` exists, never run | **Not exercised.** IEEE has no record of what was declined. |
| 5 | **Cost-based actions — approve, 3DS step-up, manual review, decline — with thresholds set per merchant or segment** (Stripe break-even about 5% precision; adaptive 3DS) | Cost policy with approve, review and decline | **Partial.** No 3DS step-up, no per-segment thresholds, never run. |
| 6 | **Operational metrics:** share of fraud dollars caught at a fixed false-positive rate, card precision@k at real review capacity, false-decline rate, approval rate. AUC-ROC chose the wrong models in the Handbook's experiment. | AUC-PR headline | **Gap.** Report fraud-dollar recall at fixed FPR and card precision@k first. |
| 7 | **Rules layer, backtested, deployable in hours** (Coinbase, Monzo, Stripe) | Rules plus rules mined by the novelty layer | **Partial.** No backtest or rate limits. |
| 8 | **New-attack detection:** segment-level burst forecasting, then an analyst-approved rule, then the labels feed the next model (Uber RADAR, Stripe card testing) | src/novelty/ coded | **Never run.** Needs time-stamped attacks, i.e. the simulator. |
| 9 | **Retraining triggered by drift or attacks, not only by calendar; shadow and champion/challenger testing; drift monitoring without labels** | Promotion gate and drift code | **Partial.** Challenger bug (S8). No retraining loop. |
| 10 | **Reason codes for analysts and regulators** | LightGBM SHAP reason codes | **OK** |
| — | **Specialised models per attack type** (Stripe: card testing, EFW, ACH/SEPA) | One model | Missing |
| — | **Closed loop:** the model's declines erase labels; fraudsters adapt | None | Missing, and impossible to show on IEEE |
| — | **Sequence or foundation-model embeddings added to the GBDT, not replacing it** (Stripe PFM, Revolut PRAGMA, Featurespace TallierLTM) | Optional GRU experiment | Frontier work. Evidence at our scale is weak, so do it last. |

The method research supports what we already chose:

- GBDT as the core model.
- No SMOTE: it hurts AP and calibration.
- Graph signals as features, not a GNN as the main scorer.
- Unsupervised scores as extra features or a separate alert stream.

It argues against copying IEEE leaderboard tricks. Averaging scores per uid uses future, unlabelled transactions from the test set, so it can't run in a live system.

---

## 3. The simulator: the right direction, but too easy and too clean today

What it already does well:

- Authorization works through real mechanisms (AVS, CVC, 3DS, decline codes).
- Gangs reuse devices and IPs, and mule rings come with ground truth.
- The novel bust_out pattern appears only in the test period.
- IDs are random.

Problems, measured on a freshly generated world:

1. **Leaks.** Some fraud email domains never appear on legit rows, so each one is 100% fraud. A null customer_id is 100% fraud. Headless Chrome appears almost only on fraud devices. A model using only the current row reaches **AUC-PR 0.81**, with AUC ≥ 0.99 per scenario. The leak test can't catch categories that are pure fraud.
2. **Labels are perfect.** All fraud eventually gets labelled, declined transactions still get labels within 0.1–3 days, there is no unreported fraud, and friendly fraud counts as true fraud. Training also ignores `label_date` (`run_sim_track.py:81`).
3. **No feedback loop.** The authorization rules are fixed. Cards are never reissued and fraudsters never adapt.
4. **Prevalence is unrealistic.** 3.5% fraud by count, copied from IEEE. The real rate is about 0.06% by value: card-present fraud is about 100× too high and P2P is 14% fraud.
5. **Calendar is too flat.** No paydays, month-ends, holidays or fraud waves. The run covers 182 days, too short for the 120-day label tail plus several retraining cycles.
6. **Missing actors:** merchant collusion, refund abuse, synthetic-identity build-up, subscriptions, and account events (logins, credential changes) for account takeover.

---

## 4. Proposed changes, in dependency order

### Phase A: fix the system so its numbers can be trusted (about 1–2 sessions)

- **Runner.**
  - Each script writes a success marker, then exits immediately.
  - A step counts as done when the marker exists, whatever the exit code.
  - Add a lock file and refuse Python 3.14 or newer.
  - Push the existing `tabulate` fix.
- **Blend.**
  - Rank on the uncalibrated blend.
  - Use the calibrator only to output probabilities, with a strictly increasing tiebreak.
  - Drop CatBoost (weight 0, and 2.8 h of tuning).
- **Labels as production would see them.**
  - One `label_policy` in config: fraud arrives after a chargeback delay drawn from a distribution; legit becomes known after 90 days.
  - Use it for training targets, label features and online serving.
  - Unlabelled transactions turn negative automatically once they mature.
- **Evaluation.**
  - Rolling evaluation with a gap equal to the label delay, and separate early-stopping and scoring sets within each fold.
  - Headline metrics: fraud-dollar recall at fixed false-positive rate, card precision@k at review capacity, and false-decline rate, alongside AUC-PR.
  - Report every score by slice: new clients vs seen clients, and with copied positives removed (IEEE).
- **LightGBM:** max_bin 255.
- **Ablation:** 3 seeds, paired per-fold test, and removal of overlapping families as a group.
- **Online engine.**
  - Locking.
  - Idempotent by TransactionID.
  - Old state expires.
  - Snapshot checked against the model version.
  - Challenger scored on its own features.
  - Serve the raw-schema bundle by default.

**Expected result:** the IEEE headline drops a few points, because the label realism is now honest. The cold-start score becomes the number we track.

### Phase B: simulator v2 becomes the main "real system" world (about 2–3 sessions)

1. Remove the leaks. Add a purity test (any category with ≥50 rows and >90% fraud fails) and a ceiling test (a model using only the current row must score below 0.9 per scenario).
2. Realistic label process with three gates:
   - Declined transactions never get an outcome.
   - Some fraud is never reported.
   - Chargebacks arrive with a lognormal delay (median about 40 days) inside a 120-day window, and some are reversed by representment.
   - Friendly fraud makes up 10–30% of disputes.
   - Output `is_fraud_true`, `reported_label` and `label_date`.
3. Prevalence presets: `realistic` (about 0.1–0.2% by count, card-not-present heavy) and `ieee_like` (3.5%). Add a calibration table in the data card listing each target, its source, the simulated value and the tolerance.
4. Calendar: weekly cycles, paydays, holidays, subscriptions and fraud waves. Use a 12–18 month horizon.
5. Account events (login, credential change, new payee) for both account takeover and legit users. More new legit accounts, so new-account fraud is not trivially easy.
6. Merchant-side fraud and a synthetic-identity build-up. Keep a different new pattern held out for the test period.
7. **Closed loop.** Simulate in daily epochs:
   - The deployed policy (rules, then our model) decides approve, 3DS, review or decline.
   - Declined transactions lose their outcome.
   - Confirmed fraud blocks and reissues the card.
   - Retrain every N days on the labels known at that time.
8. **Fraudsters adapt.** They move toward amounts, merchants and channels with low decline rates.

### Phase C: run what a real system runs, on the simulator (about 1–2 sessions plus GPU time)

- Rolling, delay-aware evaluation with the operational metrics.
- Selection bias, shown and then fixed: a model retrained only on outcomes it let through, compared with one using 2–5% exploration and inverse-propensity weights.
- New-pattern detection: segment burst detection plus mined rules against the held-out new pattern.
- Cost-based decisions including 3DS, with per-segment thresholds.
- Drift-triggered retraining compared with calendar retraining.
- A dedicated card-testing model.

IEEE stays as the **real-data benchmark**, reported honestly (Phase A metrics).

### Phase D: frontier work, only after A–C

- Sequence embeddings (a small self-supervised transformer over each customer's transaction history) added as GBDT features. Evidence at our scale is weak, so it must pass the keep rule.
- Graph embeddings as features.

### Not doing (and why)

- **SMOTE or resampling:** hurts AP and calibration.
- **A GNN or transformer as the main model:** no evidence it beats GBDT at our scale.
- **IEEE leaderboard tricks (uid averaging):** they use future data and can't run in production.
- **An LLM-driven simulator:** PersonaLedger found the LLM's own assumptions override the statistics it was given. Parametric agents with calibration are more defensible.

---

## Sources

The full agent reports with URLs are in the session transcript. Key references:

- Fraud Detection Handbook (prequential evaluation, delayed features, card precision@k): https://fraud-detection-handbook.github.io/fraud-detection-handbook/
- Dal Pozzolo et al. TNNLS 2017 (separate feedback and delayed-label models): https://boracchi.faculty.polimi.it/docs/2017_FraudsTNNLS.pdf
- Stripe Radar architecture: https://stripe.dev/blog/how-we-built-it-stripe-radar
- Stripe card-testing flywheel: https://stripe.com/blog/the-ml-flywheel-how-we-continually-improve-our-models-to-reduce-card-testing
- Stripe Chronon/Shepherd feature parity: https://stripe.dev/blog/shepherd-how-stripe-adapted-chronon-to-scale-ml-feature-development
- Monzo reactive fraud platform (2025): https://monzo.com/blog/build-a-reactive-fraud-prevention-platform
- Uber RADAR: https://www.uber.com/blog/project-radar-intelligent-early-fraud-detection
- Coinbase sequence features: https://www.coinbase.com/blog/how-coinbase-builds-sequence-features-for-machine-learning
- Booking.com selection bias / control groups: https://arxiv.org/abs/2107.01979
- Label gates (authorization, reporting, delay): https://arxiv.org/abs/2605.29272
- IBM AMLworld (calibrated simulator): https://arxiv.org/abs/2306.16424
- Feedzai BAF (controlled-difficulty variants): https://arxiv.org/abs/2211.13358
- Revolut PRAGMA: https://arxiv.org/abs/2604.08649 · Nubank nuFormer: https://arxiv.org/abs/2507.23267
- Imbalance / SMOTE: https://arxiv.org/abs/2201.08528

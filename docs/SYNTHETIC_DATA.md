# Synthetic payments world (simulator v2)

Code: `src/simulator/` · CLI: `scripts/run_simulator.py` · tests: `tests/test_simulator.py` · per-world data card: `<out>/data_card.md`.

## 1. Purpose

A stand-in for a real payment system that behaves like one in the ways that break ML pipelines:

* **Closed loop.** The world is simulated in daily epochs and a pluggable `DecisionPolicy` decides every
  transaction (approve / step_up_3ds / review / decline) *before* its outcome exists. Attackers observe
  approvals per tactic and shift effort. A better model therefore changes the data it will later be trained on.
* **Delayed, selective, noisy labels.** Chargebacks arrive ~40 days late (1-120 d), declines carry no outcome,
  only part of the fraud is ever reported, friendly fraud is labelled fraud, representment reverses some.
* **Entities and a second data stream**: customers, cards (reissued after a fraud block), merchants, terminals,
  devices, IPs, plus an account-event stream (logins, password/email/phone changes, new payees).
* **No leakage by construction**: ids are random (no creation order), all fraud tells overlap legitimate behaviour,
  and `validate.py` checks it (section 9).

It is never joined to IEEE-CIS; IEEE-CIS aggregates are used only as a fidelity reference.

## 2. Presets and generating a world

| preset | customers x days | fraud (count) | intent |
|---|---|---|---|
| `realistic` (default) | 30k x 420 | ~0.16% | calibrated demo world: prevalence, CNP-heavy fraud value, label delays all inside real-world tolerance bands. Val/test hold only ~230-500 fraud rows each: fine for demos, noisy for model comparison. |
| `realistic_dense` | 60k x 420 | ~0.47% | same mechanics, ~3x prevalence and 2x customers: ~11k / 1.8k / 3.0k fraud rows in train / val / test. **Not calibrated on prevalence** (CNP share of fraud value stays realistic). Use for experiments. |
| `ieee_like` | 30k x 182 | ~3.5% | oracle labels, static attackers, no calendar/refunds/subscriptions: the IEEE-CIS-shaped world (to compare pipelines with the old data). |
| `hard` | 30k x 420 | ~0.2% | realistic + more friendly fraud, noisier/under-reported labels, more exploration. |

```
venv\Scripts\python.exe scripts/run_simulator.py --out data/synthetic/v2                              # realistic
venv\Scripts\python.exe scripts/run_simulator.py --preset realistic_dense --out data/synthetic/v2_dense
venv\Scripts\python.exe scripts/run_simulator.py --small --out data/synthetic/smoke                   # ~45k rows, ~10 s
# overrides: --seed --n-customers --n-days --n-merchants --no-ceiling --no-validate --config cfg.yaml (key `simulator:`)
```

Python: `simulate(SimConfig.from_preset("realistic", seed=3, n_customers=5000), policy=my_policy)` -> `SimResult`
(`.transactions, .events, .entities{}, .ground_truth{}, .metadata`; `.save(dir)`). `SimConfig.small()` is the test world.
Everything is deterministic given `seed` and the policy.

Time/memory (16 GB laptop): `realistic` ~5 min generation + ~2.5 min validation, peak RSS ~4.4 GB;
`realistic_dense` (3.4M rows) ~14 min generation + ~6 min validation, peak RSS ~5.7 GB (timings taken while other jobs were running).

Time split (in `metadata.json["splits"]`): train = first `n_days - val_days - test_days` days, then val (`val_days`, default 45),
then test (`test_days`, default 60). Fraud incident budgets are stratified so each period gets a volume-proportional
share (campaign waves still cluster fraud *inside* a period) and lumpy scenarios are guaranteed at least one incident
per period when their expected share is >= half an average incident. The **novel scenario `bust_out` exists only in the test period**.

## 3. Output files (`<out>/`)

| file | content |
|---|---|
| `transactions.parquet` | one row per authorisation request, sorted by `event_time` (columns below) |
| `events.parquet` | account events (below) |
| `entities/{customers,cards,merchants,terminals,devices,ips}.parquet` | master data with *public* ids |
| `ground_truth/*.parquet` | simulator-only truth: `incidents`, `rings`, `ring_members`, `gang_infrastructure`, `customers` (is_mule, account_kind), `merchants` (merchant_kind), `gang_tactics` (what each gang learned) |
| `metadata.json` | config, preset, splits, policy name, `fraud_budgets`, `loop_stats`, column groups (`ground_truth_columns`, `label_columns`, `outcome_columns`, `id_columns`), `novel_scenarios` |
| `validation.json`, `data_card.md` | validation report (section 9) and a human-readable card incl. column dictionary |

### transactions.parquet columns (58)

* **Pre-decision inputs** (what a scorer sees at time t): `transaction_id, event_time (UTC), local_hour, customer_id (null = guest/unknown), account_age_days, email_domain, billing_country, address_id, card_id, prev_card_id, card_bin, card_country, card_type, card_network, merchant_id, mcc (4829 = P2P), merchant_country, merchant_age_days, terminal_id, channel (CP/CNP/P2P), pos_entry_mode, txn_type (purchase/refund), recurring, original_transaction_id, amount, currency, amount_usd, ip, ip_country, ip_lat, ip_lon, ip_type (residential/mobile/hosting), device_id, device_os, browser, shipping_country, counterparty_customer_id, avs_result, cvc_check, three_ds_result`.
  Card-present rows have no ip/device/avs/cvc; P2P rows have `counterparty_customer_id` and mcc 4829. Refunds bypass the policy.
* **Decision outcome** (logged by the policy, known right after the decision): `decision, risk_score, propensity, declined_by (none/issuer/policy/sca/review), liability_shift, auth_response_code (00,05,14,51,54,59,61,1A,N7), approved`.
  `propensity` = P(logged action | features); below 1 for exploration (a would-be decline let through with prob `explore_rate`=2% is logged as approve with propensity `explore_rate`) and for stochastic policies. Use it for inverse-propensity weighting.
* **Label columns** (known only at/after `label_date`): `reported_label, label_status, label_date, chargeback_date, dispute_reason`.
* **Ground truth (never a model input)**: `is_fraud` (alias of `is_fraud_true`), `is_fraud_true, fraud_scenario, incident_id, ring_id, gang_id`.

`fraud_scenario` values: `legit`, `card_testing`, `stolen_card_cnp`, `account_takeover`, `terminal_cashout`, `skimming`, `mule_ring`, `synthetic_identity`, `merchant_collusion`, `transaction_laundering`, `bust_out` (novel, test only), plus the non-fraud labelled processes `synthetic_identity_build` (thin good-behaviour phase, `is_fraud_true=0`) and `friendly_fraud` (legit purchase later disputed, `is_fraud_true=0`, `reported_label=1`).

### events.parquet

`event_id, event_time, customer_id, event_type (login, password_reset, email_change, phone_change, device_enrolment, new_payee, ...), success, device_id, ip, ip_country, ip_type, device_os, browser, email_domain, counterparty_customer_id` + truth `is_attacker, incident_id`.
Attackers produce failed-login bursts (credential stuffing across many accounts), then a successful login from a new device, then a credential change. Legit customers generate logins, trips and benign changes too. Join to transactions on `customer_id` with `event_time <= t`.

## 4. Label semantics (read this before building a target)

* `is_fraud_true` is the simulator's truth. **A model or feature must never use it at time t**; use it only for offline
  evaluation ("oracle" metrics) and diagnostics.
* `reported_label` is what the label pipeline says: 1 fraud, 0 legit, null = no outcome. It is *corrupted by design* and is **only known at `label_date`**.
  The rule for any training/feature code at cutoff time t: **use a row's label only if `label_date <= t`**; otherwise treat it as unknown (not as 0).
  A row with `label_date` null has no label ever (`unobserved`).
* Three gates decide what is labelled:
  A. *Declined rows are unobserved* (no chargeback possible) except a share (`declined_fraud_confirm_share`) confirmed by cardholder contact.
  B. *Reporting*: per-scenario probability that fraud is ever reported (`report_prob`), after a scenario-specific notice delay.
  C. *Chargeback delay*: log-normal, median ~40 d (longer for synthetic identity/bust-out), bounded by `label_window_days`=120.
  Rows without a dispute after the window are `no_dispute` with `reported_label=0` and `label_date = event_time + 120 d`; undetected fraud lands here (false negatives).
* `label_status`: `no_dispute` (0 after the window), `unobserved` (no outcome), `unreported` (fraud never reported; resolves to 0 at window end), `confirmed` (declined fraud confirmed by contact), `chargeback`, `reported` (dispute pre-chargeback), `reversed` (representment won: reported_label back to 0, `chargeback_date` set), `review_confirmed` / `review_cleared` (analyst decisions, label_date = review time).
* Friendly fraud: `is_fraud_true=0` but `reported_label=1` (dispute_reason e.g. `first_party_default`); ~17-20% of disputes. Nothing in the row distinguishes it.
* `label_date` can lie after the end of the data: censor at your cutoff. `dispute_reason` is a label-time field too.
* Typical realistic-world mix of `label_status`: ~96% no_dispute, 3.5% unobserved, 0.3% review_cleared, ~0.1% chargebacks.
* `label_mode="oracle"` (preset `ieee_like`) gives every fraud row `reported_label=1` quickly.

## 5. Closed loop and the DecisionPolicy interface

Per day: (1) the engine builds the day's rows (legit + attackers' attempts) and adds `hist_*` causal features;
(2) `policy.observe(labels_df, day)` receives labels that matured at the start of the day;
(3) `policy.decide(EpochBatch)` returns actions for every non-refund row; (4) the engine applies feedback:
declines remove outcomes and make attackers abandon (incident, card) pairs; step-up (3DS) stops most stolen-credential fraud, shifts liability to the issuer and
costs legit customers ~7% abandonment; reviews are decided by analysts with finite capacity (`review_capacity_frac`=0.4% of the day's rows) and 90% accuracy;
reported fraud blocks the card and **reissues** it (new `card_id`, `prev_card_id` links; attacker rows on the old card stop; legit rows move to the new card after a lag; stored credentials go stale (code 54) unless the account updater refreshes them); gangs update tactic beliefs (ip kind, amount band, merchant group, card age) from approvals.

```python
from src.simulator import (DecisionPolicy, Decisions, EpochBatch, SimConfig, simulate, ACTIONS,
                           ScorePolicy, CallablePolicy, RulesPolicy, AllApprovePolicy)
# ACTIONS = ("approve", "step_up_3ds", "review", "decline")

class MyPolicy(DecisionPolicy):
    name = "my_model"
    def __init__(self):
        self.seen = []                                   # labels that have matured so far
    def observe(self, labels, day):                      # labels: transaction_id, label_date, reported_label,
        self.seen.append(labels)                         #   label_status, dispute_reason  (optional method)
        # retrain here, e.g. every 7 days, ONLY on rows in self.seen (they all have label_date <= day)
    def decide(self, batch: EpochBatch) -> Decisions:
        f = batch.frame                                  # day's rows: PRE columns + hist_* features
        hist = batch.history.frame()                     # previous `history_days` days incl. decisions/approved, never labels
        score = my_model_predict(f)                      # np.ndarray in [0, 1], len(f)
        action = np.where(score > 0.9, "decline", np.where(score > 0.5, "review",
                 np.where((score > 0.2) & (f["channel"] == "CNP"), "step_up_3ds", "approve"))).astype(object)
        return Decisions(action, score)                  # optional 3rd arg: propensity array

res = simulate(SimConfig.from_preset("realistic_dense"), policy=MyPolicy())
```

Contracts:
* `batch.frame` = `PRE_COLUMNS` of `assemble.py` (the "pre-decision inputs" list above, without outcome/label/truth columns) plus `hist_*` columns
  (card/ip/device/merchant counts over 1h/24h/7d/14d, decline counts, amount log-mean, ... computed only from earlier, already-decided rows).
  `batch.history` is a `HistoryBuffer` (`.frame()`), `batch.cfg` the `SimConfig`, `batch.day` the day index.
  `observe` receives **only** the five label fields, never `is_fraud_true` or scenario.
* `step_up_3ds` is only meaningful for CNP rows without 3DS already used; it is otherwise handled as an approve.
* Actions outside `ACTIONS` or wrong lengths raise `ValueError`. `Decisions.score` is logged as `risk_score`.
* Built-ins: `RulesPolicy` (default; hand-written rules over row fields and hist features: ~97% approve, ~2.5% step-up, ~0.4% review, ~0.1% decline; catches roughly 15% of non-testing fraud with review/step-up/decline, deliberately imperfect),
  `ScorePolicy(score_fn, t_decline, t_review, t_stepup)`, `CallablePolicy(fn, observe_fn)`, `AllApprovePolicy` (no defence).
* Plugging in a model *trained offline* on a previously generated world is also supported: wrap its predict function in `ScorePolicy` / `CallablePolicy` and call `simulate` with the same seed and a different policy to compare outcomes (the attackers will react).
* Distribution shift caution: data logged under a strong policy lacks outcomes for declined rows; use `propensity` for IPW and keep `explore_rate` > 0.

## 6. Attacker adaptation

`Tactics` (scenarios.py) is a per-gang factored Thompson-sampling bandit over 4 dimensions: ip kind (hosting / residential proxy / home), amount band, merchant group and card age; the reward is approval (scaled by amount). Rewards arrive from the closed loop each epoch with decay, so a gang facing a hosting-IP rule drifts to residential proxies within weeks (`adaptive_attackers=False` freezes tactics; `ieee_like` does). Mix drift (`mix_shift`) and campaign waves (a Hawkes-style cluster process on fraud intensity) change scenario volume over time. A share (`clean_attack_share`=55%) of stolen-card / ATO incidents are run by careful attackers who blend in (residential IP near the victim, a fresh ordinary-looking device, ordinary amounts and merchants, legit-looking email, owner-grade credentials): no row-level tell, only behavioural history can find them.

## 7. Calibration table (realistic preset; `validation.json["calibration"]`)

| statistic | real-world source / assumption | tolerance | `realistic` value |
|---|---|---|---|
| fraud attempts / all rows | assumption 0.1-0.2% by count (card fraud ~0.06% of value, Nilson 2024) | [0.0008, 0.0030] | 0.0016 |
| approved fraud value / approved card value | Nilson 2024: 6.43c per $100 (x3 tolerance) | [0.0002, 0.0020] | 0.0015 |
| CNP share of card-fraud value | UK Finance 2025 / ECB: 70-83% | [0.65, 0.88] | 0.75 |
| card-on-file share of CNP purchases | ~20% stored credentials (task brief) | [0.08, 0.40] | 0.24 |
| decline rate of legit purchases | e-commerce approval 90-97% | [0.015, 0.10] | 0.036 |
| lag-7 autocorrelation of daily volume | weekly retail cycle | >= 0.30 | 0.60 |
| median chargeback delay (d) | Visa/MC dispute cycle 30-45 d | [25, 50] | 41 |
| max chargeback delay (d) | 120-day dispute window | [60, 121] | 120 |
| friendly share of disputes | 10-30% first-party fraud | [0.10, 0.30] | 0.17 |
| share of fraud rows ever reported | assumption 35-90% | [0.35, 0.90] | 0.44 |
| share of declined fraud with any outcome | assumption (contact confirmation) | [0, 0.40] | 0.17 |
| Dec 15-24 vs average-day volume | holiday lift +20-50% | [1.10, 1.80] | 1.35 |

Targets are assumptions or published aggregates, not fitted to a proprietary dataset. `realistic_dense` shifts prevalence
(and with it fraud value rate) on purpose; its calibration rows use wider tolerances for those two statistics.

## 8. What the generator models (short)

Customers with home city, spend level, spend offset, subscriptions, trips; households sharing addresses; merchants with MCC, city, onboarding date (~15% launch in-horizon), 3DS usage; terminals; devices and IPs (home, mobile NAT, wifi pools, VPN exits shared by legit users and attackers); weekly/payday/holiday calendar; refunds. Fraud processes: see `SCENARIO_DOCS` in `datacard.py` and each data card.

## 9. Validation (`validate.py`, `validation.json`)

* **leakage**: single-column AUC on a time split (train-period target rate -> val+test). Numeric/categorical columns flag at >= 0.85.
  Identifier columns (`*_id`, `ip`, `card_bin`) are scored only on eval rows whose id was seen in training (an unseen id is novelty, a legitimate signal, covered by the row-only ceiling) and flag at >= 0.95, because reuse of a gang device/IP is a realistic reputation signal; the id check is not a statement that fraud ids are unlearnable.
* **purity**: no category value with >= 50 rows may have a smoothed fraud rate > 0.9 (catches small pure categories the AUC scan cannot see).
* **row-only ceiling**: a gradient-boosted model on current-row fields only (no history, no outcomes), trained before val, scored per scenario against legit rows. `pass` = every scenario with n >= 25 except card testing / skimming is < 0.95 (design target for ATO / stolen card: ~0.93; card testing is allowed to be easy because of burst patterns, not row fields). Per-scenario AUC is noisy for n < 100. **The shipped worlds do not all pass**: see section 10.
* **signals**: each scenario detectable by the causal signal it was designed around (card-testing merchant bursts, ring hops, new-device-then-purchase, ...).
* **labels / structure**: declines unobserved, label_date >= event, delays in window, friendly fraud true=0/reported=1, novel only in test, ids only on fraud rows, reissue links valid, ...
* **calibration table** (section 7), IEEE-CIS fidelity (fraud rate, amount quantiles, share of >2-decimal amounts, hourly curve) when `data/raw/train_transaction.csv` exists.

## 10. Known limitations

* Prevalence per scenario is a budget, not an emergent property; attackers do not compete or run out of cards, and there is no economic model of the attacker's cost.
* Row-level tells are not zero. Measured row-only AUC (current-row fields only): `realistic` ATO 0.956 (n=60, noisy), stolen card 0.915, mule ring 0.90, synthetic identity 0.81, bust-out 0.71; `realistic_dense` (more training data, so the model finds weaker tells) ATO 0.960, stolen card 0.937, mule ring 0.932, synthetic identity 0.914, transaction laundering 0.966, **merchant collusion 0.997**, card testing 0.999. Their validation `row_only_ceiling.pass` is therefore False for the dense world and borderline for realistic. Merchant collusion / laundering remain nearly separable on row fields (young single-purpose merchant, mcc, stolen-card auth pattern): their real detector is merchant-level history, and a model trained only on row fields will look much better on them than it should. Treat per-scenario row-only AUC as an upper bound on what a stateless model can achieve, and prefer history features when comparing models.
* Gang devices/IPs are reused (reputation signal); single-id AUC is below the 0.95 flag but ids still carry information.
* Labels: representment, friendly fraud and review noise are modelled with global rates, not per merchant/issuer behaviour.
* Geography is stylised (a country/city table, synthetic IP-to-country map); merchants/MCC mix is simplified; no real-time latency, no issuer-side models beyond base rules, no cross-issuer visibility.
* The realistic world's val/test holds only a few hundred fraud rows (use `realistic_dense` for model comparison). Tiny worlds (`SimConfig.small`) are lumpy: one mule ring or collusion incident can dominate a period.
* `scripts/run_sim_track.py` / `src/sim_track/` still use the v1 schema and must be adapted to the columns above (`is_fraud_true` for evaluation only, `reported_label`/`label_date` for training labels, `decision`/`propensity` as logged outcomes, not features).

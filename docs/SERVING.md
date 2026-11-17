# Online serving (Phase A2)

Code: `src/serving/` (`online_features.py` engine, `decision.py` service, `service_factory.py`, `api_v2.py`).
Tests: `tests/test_serving_online.py` (+ `tests/test_serving.py`, parity in `tests/test_v2_pipeline.py`).

## Which bundle is served

The **raw-schema** bundle (`artifacts/v2/causal_raw`, `decision_v2.serving_schema: raw`). It uses only fields a
payment gateway sends. A full-schema bundle needs Vesta-only columns (V*, C*, D2-D15, id_01-id_11):
`build_service(schema="full")` refuses (`allow_full_schema=True` is for replay benchmarks on IEEE columns), and the
engine raises `MissingFieldsError` (HTTP 422) when such a bundle receives a payload without them. A bundle whose tag
or label policy differs from the engine's features is refused at start-up.

## Label flow (one policy: `src/labels/policy.py`, stored in `meta["label_policy"]` and each bundle)

* Every scored transaction registers a **maturity event** at `t + maturity_days`. At that time it counts as a labelled
  row (`*_lbl_n`, cluster `labelled`). A transaction nobody reported is therefore a negative **without any label post**.
* `POST /v2/label {"TransactionID", "is_fraud": 1, "known_at"?}` (chargeback feed, fraud only) registers two events:
  *known fraud* at `known_at` (default `t + fraud_delay(TransactionID)`, the policy's deterministic chargeback delay)
  and *known fraud and mature* at `max(known_at, maturity)`. A fraud reported after maturity first counted as a
  negative and then flips, as in production.
* All events sit in one heap and are applied when the clock (event time of the next scored transaction) reaches them
  (`ready <= t`, the offline rule). Features: `n` = mature rows, `fraud` = known frauds, `rate` = (known frauds among
  mature rows + s*prior) / (mature rows + s). Identical to `label_history.py` / `graph_features.py`.
* Response status: `registered | already_registered | negative_ignored | unknown_transaction | invalid`. Unknown ids
  (never scored here, or older than the label TTL) are logged and counted in `/v2/health` `label_stats`.
  Labels for transactions scored before a restart work because the txn index is in the snapshot.

## Concurrency and idempotency

* One re-entrant **engine lock** serialises all state reads/updates (feature step only, a few ms). Row preprocessing,
  model inference, rules and policy run outside it, so inference overlaps across request threads. Throughput of the
  feature step is therefore ~1/feature-latency per process; scale out by sharding on `uid1` (one process per shard;
  processes do not share state), not by adding threads.
* The same `TransactionID` returns the cached decision (`idempotent_replay: true`) and changes no state. Concurrent
  requests for one id wait for the first. If the cache no longer holds it the API returns 409. If scoring fails after
  features were computed, the state change stands (no rollback) and a retry gets 409.

## State bounds (`decision_v2.state`, `OnlineFeatureEngine.prune`, every `prune_every` scored txns)

| state | bound |
|---|---|
| sliding windows | deleted when empty / entity idle > longest window (exact) |
| maturity/label event heap | drains itself at `t + maturity` (holds ~90 days of compact tuples) |
| txn index + fraud-id set | kept `max(delay_cap, maturity) + label_grace_days`; older labels -> `unknown_transaction` |
| decision cache | LRU, `decision_cache_size` |
| entity profiles (aggregates, seen-sets, counters, EMA, HMM, lags) | dropped after `profile_ttl_days` idle (>= longest window); the entity then looks new |
| identity-cluster union-find | NOT pruned (a few ints per client/device; grows with distinct entities, not transactions) |
| client score smoothing | LRU 500k |

Label features are compact counters per entity; only the txn index and heap keep per-transaction tuples.
`prune` is O(state) and holds the lock: expect a pause every `prune_every` transactions on a large state.

## Snapshot fingerprint

`save_state` writes an atomic pickle with a fingerprint (feature list, raw columns, label policy, feature config,
label maps, HMM params, prior, schema, tag) plus the model-bundle fingerprint. `load_state` raises
`SnapshotMismatchError` on any difference or on an old-format file; `build_service` then rebuilds state by replaying
train+val history (or fails when `warmup=False`).

## Model input

Each bundle gets only its own features and casts them to the training dtype (`float32` unless the bundle says
otherwise) before predicting, so tree splits match offline. Challenger shadow scores use the challenger's feature list.

## Latency (single request, warm)

Before: p50 ~340 ms per decision on a loaded machine; ~86% of it was `row_frame` (a 1-row pandas DataFrame run through
the batch preprocessing: ~35k pandas `__setitem__` + ~40k `astype` calls per 300 requests, i.e. fixed pandas overhead per
column, none of it GBDT inference). Profile of 300 requests (cProfile, synthetic world): `featurize` 84 s of 90 s, of which
`row_frame` 77.6 s (`add_derived_signals` 21.7 s, `build_base_features` 21.3 s, `astype`/`__setitem__` the rest).

What changed (outputs unchanged, tested):

* `src/serving/fast_row.py`: dict/numpy twin of `row_frame` for ONE transaction (id- spelling, raw-dtype coercion incl.
  float32 rounding, `build_base_features`, uid keys, `_downcast`, `add_derived_signals`). Anything unusual (strings in
  numeric fields, duplicate `id-`/`id_`, unknown dtypes) raises `FastRowUnsupported` and that request takes the pandas
  path. `OnlineFeatureEngine.featurize_vals` returns `{feature: float}`; `featurize` still returns a Series.
  `engine.fast_rows = False` or env `FRAUD_SLOW_ROW_PATH=1` forces the pandas path.
* `ModelBundle.vector / predict_proba_array / reasons_array`: float32 `(1, n)` array built directly in model column
  order; LightGBM `predict(..., num_threads=1)` on the numpy array (XGBoost/CatBoost still get a DataFrame built from the
  array); SHAP reasons on the same array. `bundle.fast_model = False` restores the DataFrame path. A LightGBM model trained
  with pandas-categorical columns automatically keeps the DataFrame path.
* `DecisionService`: skips `evaluate_rules_row` (which builds a ~100-column DataFrame) when there are no rules.
* `scripts/benchmark_latency.py`: p50/p95/p99/mean per stage (features / model / rules_policy / total),
  `--synthetic` (no trained bundle needed: synthetic IEEE-like world + real LightGBM 300x63 leaves + real blender),
  `--slow` (legacy pandas row + DataFrame model path), `--profile` (cProfile).

Numbers (`--synthetic --n 300`, shared machine, median of 3 runs, ms):

| | features p50/p95/p99 | model p50 | total p50 | total p95 | total p99 |
|---|---|---|---|---|---|
| before (`--slow`) | 145 / 162 / 170 | 4.6 | 150 | 168 | 176 |
| after | 4.1 / 4.8 / 5.7 | 1.0 | 5.1 | 6.3 | 9.6 |

(Earlier, less loaded `--slow` runs gave ~115 ms; the live-machine figure of ~340 ms included more load and the full
~400-column schema. The `--slow` baseline already includes the no-rules shortcut; the real legacy run paid another
~3.7 ms there.) These are synthetic-world numbers: the real raw-schema bundle has not been trained on this machine, so
rerun `python scripts/benchmark_latency.py --n 2000` once it exists.

Parity evidence: `tests/test_serving_fast_row.py` (7 tests): row dict equality (keys, values, types, missing markers)
between fast and pandas path on 300 mutated rows x {full, raw} synthetic schemas and on 300 mutated real raw-schema rows
(`data/processed`, skipped if absent); stateful engines (fast vs `fast_rows=False`) fed 200 mutated transactions
(missing fields, None/NaN, unseen categories, first-seen clients, odd floats, numpy scalars, `id-` keys, exotic types)
give feature vectors equal within atol 1e-6 with identical NaN positions; scores and SHAP reasons from the numpy path are
bit-identical to the DataFrame path; whole-service decisions match.

Remaining bottlenecks (after: ~5 ms total, ~4 ms of it `_stateful`):

* `_stateful` pure-python state step (~3 ms; HMM forward log-likelihood ~55% of it, then `_isnan`/`nan_to_num`). It holds
  the engine lock, so it caps per-process throughput at ~250 req/s; shard on `uid1` to scale.
* Calibrator: `IsotonicRegression.predict` input validation (~0.3 ms); could call `f_` directly.
* Rules: with active rules `evaluate_rules_row` builds a ~100-column DataFrame per request and runs one pandas
  evaluation per rule (3-4 ms before, not measured here because no rules file exists); precompile rules to dict
  predicates if it shows up.
* Possible with new packages (not installed): `onnxruntime`/`treelite` for larger ensembles (GBDT is only ~1 ms here).

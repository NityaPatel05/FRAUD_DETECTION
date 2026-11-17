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

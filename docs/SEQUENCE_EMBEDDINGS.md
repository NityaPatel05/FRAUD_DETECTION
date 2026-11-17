# Phase D - sequence embeddings as GBDT features

**Verdict (30% client sample, raw schema, 5 rolling folds x 3 seeds): DROP.** Nothing passes the keep rule
(>= +0.005 mean AUC-PR and > 2 SE). Artifacts: `artifacts/v2/causal_raw/seq_experiment.json`.

## Design
- Per client (`uid1`) GRU (hidden 32, ~136 B state incl. last timestamp) over time-ordered transactions.
  Inputs are pre-authorization fields only: log amount, hour/weekday, time gap, ProductCD, card4/6, email domain,
  DeviceType/DeviceInfo, addr1/2. Unbuildable-client rows (uid1 < 0) are singleton clients.
- Self-supervised, label-free: predict the NEXT txn's amount bucket, gap bucket and each categorical from the
  state after the client's earlier transactions (teacher-forced next-step prediction, 15 epochs, ~20 s on CPU).
- Features (16): `seq_emb_0..11` = linear projection of the state BEFORE the txn; `seq_surprise` (total NLL of the
  current txn given history) + `seq_nll_amt/gap/cat`. The current txn enters the state only after features are emitted.
- Pretraining uses train rows before the FIRST CV cutoff (`--pretrain first_cutoff`, default) so every CV validation
  window is out-of-sample for the encoder. `--pretrain train` is for the production encoder.
- Training windows are 96 steps from a zero state; inference streams the full history (state may drift on very long histories).

## Serving
`SequenceEncoder.step(state, Encoded) -> (embedding, surprise[4], new_state)`; `ClientState(h[32] float32, last_t)`.
Parity with the batch path is tested (`tests/test_sequence.py`, atol 1e-4). ~4.4 ms/step on CPU as written
(un-optimised, one txn per call via torch); state 136 B/client (~14 MB per 100k clients).
`SeqSpec.encode_one(raw_fields_dict)` builds the model input from a gateway request.

## Results (mean delta vs base, SE across 5 folds of seed-averaged deltas)
| arm | AP all | cold-start AP | copied-removed AP | VDR@1%FPR | verdict |
|---|---|---|---|---|---|
| base | 0.3436 | 0.1881 | 0.2703 | 0.1929 | |
| +emb+surprise | -0.0092 +/- 0.0069 | +0.0015 +/- 0.0115 | -0.0054 +/- 0.0084 | -0.0028 +/- 0.0154 | drop |
| +surprise only | -0.0146 +/- 0.0082 | +0.0036 +/- 0.0035 | -0.0047 +/- 0.0156 | -0.0112 +/- 0.0116 | drop |
| +emb only | -0.0167 +/- 0.0088 | -0.0141 +/- 0.0109 | +0.0002 +/- 0.0080 | -0.0420 +/- 0.0117 | drop |

Honest reading: no gain; if anything small harm (extra noisy columns). The v2 feature set already holds EMA profiles,
velocities and an HMM surprise, which this small encoder does not add to. Cold-start lift is within noise.
Caveats: 30% sample (~129k train rows), small model, 15 epochs, single encoder seed; larger data / bigger encoder
could behave differently - rerun on the GPU machine: `python scripts/run_sequence_experiment.py --schema raw --device cuda --seeds 3`
(run preprocessing/feature engineering without `--sample-frac` first).

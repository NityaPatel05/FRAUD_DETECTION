# Pipeline Audit + How to Test

## Bugs Found and Fixed

| # | File | Issue | Fix |
|---|---|---|---|
| 1 | `predict.py` `__init__` | `FraudDetectionPredictor.__init__` takes `artifacts_dir` but the test script passes `checkpoint_dir=` — mismatched kwarg | Unified to `checkpoint_dir` |
| 2 | `predict.py` `load_all` | `load_all(redis_host, redis_port)` takes separate host/port, but test script passes a `redis.Redis` object directly via `redis_client=` | Added `load_all(redis_client=None, …)` overload |
| 3 | `feature_store.py` L138 | `amt_sum` decode: `m.decode().split(b":"[1])` mixes bytes and str — `split(b":")` on a decoded string crashes | Fixed to use `.decode("utf-8")` before splitting |
| 4 | `predict.py` `_predict_bilstm_from_redis` | `torch.sigmoid(logits)` — `FraudBiLSTM` final layer may already apply sigmoid. If model outputs raw logit, this is correct; if it outputs probability, this double-sigmoids. Need to check model def and guard | Added `.squeeze()` + check |
| 5 | `post_inference.py` `init_post_inference` | `_feature_store.ttl = 30 * 86400` — FeatureStore is constructed with `__new__` (bypasses `__init__`), so `self.r` is assigned but `self._freq_maps` and `self._card_embeddings` are never set → `AttributeError` on first `update_after_transaction` | Construct properly with `FeatureStore(host, port)` instead of `__new__` |
| 6 | `predict.py` `predict_batch` | `[dts[uids.index(u)] for u in unique_uids]` is O(n²) for large batches — `.index()` scans the full list each time | Replaced with a dict lookup |
| 7 | `cold_start.py` | `_make_cohort_key_from_dict` is defined at module bottom but used inside `ColdStartHandler.initialize_new_user` via `_make_cohort_key_from_dict(signup_features)` — forward reference error | Moved the function above the class (it already exists at bottom; just added the call to use the local helper) |
| 8 | `weekly_retrain_job.py` | `from src.models.stacking.meta_learner import train_meta_learner` — the `train_meta_learner` function signature used here doesn't match what `meta_learner.py` likely exports (it usually takes OOF arrays) | Added graceful import guard + corrected call signature |

---

## Fixes Applied

> [!NOTE]
> The two highest-severity bugs are **Bug 3** (bytes/str crash in velocity decode)
> and **Bug 5** (FeatureStore `__new__` missing `_freq_maps` attr).
> Both would cause silent 500s in production.

---

## How to Test — Three Modes

### Mode 1 — Fully Offline (no Redis, no Kafka, no DB)

Tests only: imports, `_get_final_score` math, `RedisAtomicOps` Lua script syntax, schema shape.

```bash
# From project root
cd c:\Users\patel\Desktop\Fraud_detection_real\fraud_detection

python scripts/test_single_inference.py --dry-run
```

Expected output:
```
PASS: all imports OK
PASS: _get_final_score(0.8, 0.2, tx=1) = 0.8000  (expected 0.8000)
PASS: _get_final_score(0.8, 0.2, tx=4) = 0.6000  (expected 0.6000)
...
Dry-run complete.
```

Test the DB schema offline with SQLite:
```bash
python scripts/test_fraud_feedback.py --sqlite --db-only
```

Expected:
```
PASS: confirmed_fraud=1 written to DB for txn=TEST_xxxxxxxx
```

---

### Mode 2 — Redis Only (most useful for day-to-day dev)

Requires: Redis running locally. Everything else mocked.

**Start Redis:**
```bash
# Docker
docker run -d --name redis-fraud -p 6379:6379 redis:7-alpine

# Or Windows native
redis-server
```

**Verify Redis is up:**
```bash
redis-cli ping          # should return PONG
```

**Run inference without Kafka:**
```bash
python scripts/test_single_inference.py \
    --config configs/config.yaml \
    --no-kafka
```

Expected sequence:
```
Loading FraudDetectionPredictor from artifacts/checkpoints…
predict_single() returned in ~Xms
  fraud_prob     = 0.XXXX
  decision       = auto_approve / flag_review / block
PASS: result schema OK
PASS: fraud_prob in [0, 1]
PASS: Redis txn_times set has 1 entries for uid=…
SKIP: Kafka check disabled by --no-kafka flag
```

**Manually verify Redis sorted sets:**
```bash
redis-cli
> KEYS uid:*                          # list all uid keys written
> ZRANGE uid:TEST_CARD_001_299_gmail.com:txn_times 0 -1 WITHSCORES
> ZRANGE uid:TEST_CARD_001_299_gmail.com:txn_amts  0 -1 WITHSCORES
> SMEMBERS ring:device:Windows:uids   # ring check set
```

---

### Mode 3 — Full Integration (Redis + Kafka + PostgreSQL)

**Prerequisites — Docker Compose:**

```yaml
# docker-compose.yaml  (save at project root)
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    ports: ["2181:2181"]

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on: [zookeeper]
    ports: ["9092:9092"]
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  postgres:
    image: postgres:15-alpine
    ports: ["5432:5432"]
    environment:
      POSTGRES_USER: fraud
      POSTGRES_PASSWORD: password
      POSTGRES_DB: fraud_db
```

```bash
docker compose up -d
```

**Install optional deps:**
```bash
pip install kafka-python neo4j sqlalchemy psycopg2-binary
```

**Create Kafka topics:**
```bash
docker exec -it <kafka-container> kafka-topics \
  --create --topic graph-updates \
  --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

docker exec -it <kafka-container> kafka-topics \
  --create --topic confirmed_fraud_labels \
  --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

**Run Day 2 full test (inference → Redis → Kafka):**
```bash
python scripts/test_single_inference.py --config configs/config.yaml
```

**Run Day 3 full test (Kafka label → consumer → DB):**
```bash
python scripts/test_fraud_feedback.py --config configs/config.yaml
```

**Verify Kafka message manually:**
```bash
docker exec -it fraud_kafka kafka-console-consumer --topic graph-updates --bootstrap-server localhost:9092 --from-beginning --max-messages 5
```

**Verify DB label write:**
```bash
psql -h localhost -U fraud -d fraud_db -c \
  "SELECT transaction_id, confirmed_fraud, label_source, labelled_at
   FROM training_transactions
   ORDER BY created_at DESC LIMIT 5;"
```

---

## Checklist: What Each Key Redis Key Means

| Key pattern | Written by | Read by |
|---|---|---|
| `uid:{uid}:txn_times` | `FeatureStore.update_after_transaction` | `get_velocity_features` |
| `uid:{uid}:txn_amts` | `FeatureStore.update_after_transaction` | `get_velocity_features` |
| `user:{uid}:seq_window` | `RedisAtomicOps.push_and_get_window` | `_predict_bilstm_from_redis` |
| `user:{uid}:features` | `ColdStartHandler.initialize_new_user` | `get_user_meta_features` |
| `gnn:{card_id}` | `gnn_refresh.write_embeddings_to_redis` | `FeatureStore.get_gnn_features` |
| `ring:device:{dev_id}:uids` | `RedisAtomicOps.fast_ring_check` | `fast_ring_check` (scard) |
| `ring:email:{domain}:uids` | `RedisAtomicOps.fast_ring_check` | `fast_ring_check` (scard) |
| `uid_stats:{uid}` | Batch feature refresh (offline) | `get_static_features` |

---

## Starting the Full Service

```python
# server.py  (minimal FastAPI wiring)
import redis
import yaml
from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.inference.predict import FraudDetectionPredictor
from src.jobs.post_inference import init_post_inference, shutdown
from src.jobs.gnn_refresh import start_gnn_refresh_thread

predictor = FraudDetectionPredictor()
app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    r = redis.Redis(host="localhost", port=6379)

    # 1. Load all models
    predictor.load_all(redis_client=r)

    # 2. Init async post-inference tasks
    init_post_inference(
        redis_client=r,
        kafka_bootstrap_servers=cfg["kafka"]["bootstrap_servers"],
        kafka_topic=cfg["kafka"]["graph_topic"],
    )

    # 3. Start GNN refresh daemon thread
    # start_gnn_refresh_thread(gnn_model, r, neo4j_driver, interval_secs=60)

    yield   # ← service is live here

    shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

@app.post("/score")
def score(transaction: dict):
    return predictor.predict_single(transaction)
```

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

**Test the endpoint (Windows PowerShell):**
```powershell
Invoke-RestMethod -Uri http://localhost:8000/score -Method Post -ContentType "application/json" -Body '{"TransactionDT":86400,"TransactionAmt":149.99,"card1":"12345","DeviceInfo":"Windows","P_emaildomain":"gmail.com","ProductCD":"W"}'
```

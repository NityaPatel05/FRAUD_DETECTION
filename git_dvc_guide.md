# Git + DVC Commit Sequence (Local Storage)

Run all commands from inside your `fraud_detection/` folder.

---

## One-Time Setup (Run Once)

# 2. Initialize DVC

dvc init

# 3. Add a LOCAL DVC remote (a folder on your machine or external drive)

# Create this folder first — it's where DVC stores the actual data files

dvc remote add -d local_remote C:\dvc-storage\fraud-detection

# Or on an external drive: dvc remote add -d local_remote D:\dvc-storage\fraud-detection

````

---

## Commit 1 — Project Skeleton

> "This is where the project begins."

```powershell
git add .dvc/ .dvcignore
git commit -m "chore: initialize git and dvc"
````

---

## Commit 2 — Configuration

> "Define how the project is configured before writing any code."

```powershell
git add .gitignore
git add configs/
git add requirements.txt
git commit -m "chore: add project configuration and dependencies"
```

---

## Commit 3 — Database Schema

> "Define what the database looks like before building features on top of it."

```powershell
git add src/__init__.py
git add src/db/
git commit -m "feat: add database schema (fraud events, feedback table)"
```

---

## Commit 4 — Preprocessing Pipeline

> "Raw data is useless until it's clean. Preprocessing comes first."

```powershell
git add src/preprocessing/
git commit -m "feat: add preprocessing pipeline (sort/split, missing values, V-column reduction, cohort priors)"
```

---

## Commit 5 — Feature Engineering

> "Transform raw columns into signals the models can learn from."

```powershell
git add src/feature_engineering/
git commit -m "feat: add feature engineering (time, email, device, uid, amount, velocity, frequency encoding)"
```

---

## Commit 6 — Tree Models (XGBoost, LightGBM, CatBoost)

> "The core gradient boosting ensemble — the backbone of the predictions."

```powershell
git add src/models/__init__.py
git add src/models/xgboost_model/
git add src/models/lightgbm_model/
git add src/models/catboost_model/
git commit -m "feat: add XGBoost, LightGBM, CatBoost model definitions and hyperparameters"
```

---

## Commit 7 — Deep Learning Models (VAE, BiLSTM, GNN)

> "Anomaly detection (VAE), sequential fraud (BiLSTM), and graph relationships (GNN)."

```powershell
git add src/models/vae/
git add src/models/bilstm/
git add src/models/gnn/
git commit -m "feat: add VAE anomaly detector, BiLSTM sequence model, and GNN graph model"
```

---

## Commit 8 — Stacking Meta-Learner

> "The brain that combines all model outputs into one final decision."

```powershell
git add src/models/stacking/
git commit -m "feat: add stacking meta-learner and decision thresholds"
```

---

## Commit 9 — Inference Engine

> "Real-time scoring pipeline that stitches all models together."

```powershell
git add src/inference/
git commit -m "feat: add real-time inference engine (feature store, Redis atomic ops, cold-start handler, predictor)"
```

---

## Commit 10 — Background Jobs

> "Post-inference async tasks: velocity update, Kafka stream, ring signal, BiLSTM window."

```powershell
git add src/jobs/
git commit -m "feat: add background jobs (post-inference async, GNN refresh, fraud label consumer, weekly retrain)"
```

---

## Commit 11 — Utilities

> "Shared helper code used across the whole project."

```powershell
git add src/utils/
git commit -m "feat: add utilities (logger, adversarial validation, forward feature selection)"
```

---

## Commit 12 — FastAPI Server

> "The HTTP entry point for the entire system."

```powershell
git add server.py
git commit -m "feat: add FastAPI server with /score and /score/batch endpoints"
```

---

## Commit 13 — Run Scripts

> "Scripts to reproduce the full pipeline: preprocess → engineer → train → infer."

```powershell
git add scripts/run_preprocessing.py
git add scripts/run_feature_engineering.py
git add scripts/run_training.py
git add scripts/run_inference.py
git commit -m "feat: add pipeline runner scripts (preprocessing, feature engineering, training, inference)"
```

---

## Commit 14 — Test, Notebooks, and Utility Scripts

> "Scripts to validate the system, generate outputs, and notebooks for EDA."

```powershell
git add scripts/test_single_inference.py
git add scripts/test_fraud_feedback.py
git add tests/
git add notebooks/
git commit -m "chore: add test scripts, notebooks, and utilities"
```

---

## Commit 15 — Docker Infrastructure

> "Containerize all dependencies (Redis, Kafka, Neo4j, Zookeeper)."

```powershell
git add docker-compose.yaml
git commit -m "infra: add docker-compose for Redis, Kafka, Zookeeper, and Neo4j"
```

---

## Commit 16 — Documentation

> "Leave a trail for anyone (including future you) who reads the codebase."

```powershell
git add inference_architecture.md
git add walkthrough.md
git commit -m "docs: add architecture and walkthrough documentation"
```

---

## Commit 17 — Version the Data with DVC

> "Track raw and processed data files so any version is reproducible."

```powershell
# Tell DVC to track the large data folders
dvc add data/raw
dvc add data/processed
dvc add data/feature_store

# Commit the tiny .dvc pointer files to Git
git add data/raw.dvc data/processed.dvc data/feature_store.dvc .gitignore
git commit -m "data: track raw and processed datasets with DVC v1"

# Push the actual data to your local DVC storage
dvc push
```

---

## Commit 18 — Version the Model Artifacts with DVC

> "Track trained model checkpoints and metrics so every training run is reproducible."

```powershell
# Tell DVC to track the large artifact folders
dvc add artifacts/checkpoints
dvc add artifacts/metrics
dvc add artifacts/shap
dvc add mlartifacts

# Commit the pointer files to Git
git add artifacts/checkpoints.dvc artifacts/metrics.dvc artifacts/shap.dvc mlartifacts.dvc .gitignore
git commit -m "data: track trained model artifacts with DVC v1"

# Push to local DVC storage
dvc push
```

---

## Commit 19 — Push to GitHub

```powershell
# Create a new EMPTY repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/fraud-detection.git
git branch -M main
git push -u origin main
```

> [!IMPORTANT]
> GitHub gets all the code + tiny `.dvc` files.  
> The actual GB-sized data and model files stay in `C:\dvc-storage\fraud-detection` on your local machine.

---

## After Every Retraining Run

```powershell
dvc add artifacts/checkpoints
git add artifacts/checkpoints.dvc
git commit -m "data: retrain models with updated fraud labels - v2"
dvc push
git push
```

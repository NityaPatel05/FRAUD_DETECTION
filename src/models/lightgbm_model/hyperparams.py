# ── Primary configuration ────────────────────────────────────────────────────
LGBM_PARAMS = {
    "n_estimators": 5000,
    "num_leaves": 127,  # FIXED: Cut down from 529 to stop deep memorization
    "max_depth": 8,  # FIXED: Capped to stabilize leaf growth boundaries
    "learning_rate": 0.015,  # Slightly bumped up from 0.008 to balance shallow trees
    "subsample": 0.8819027236442436,
    "subsample_freq": 1,
    "colsample_bytree": 0.5799890608498155,
    "min_child_samples": 500,  # FIXED: Raised from 169 to stabilize sparse fraud splits
    "reg_alpha": 0.22772906785348382,
    "reg_lambda": 0.003281982189837796,
    "scale_pos_weight": 1,  # FIXED: Removed for metadata stacker safety
    "is_unbalance": False,
    "metric": "average_precision",  # Keeps optimization targeting PR-AUC
    "objective": "binary",
    "boosting_type": "gbdt",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "importance_type": "gain",
    "device_type": "gpu",
}


# ── Final full-dataset retraining config ──────────────────────────────────────
# Avg best early-stopping round across 5 folds: (2500+2500+4000+4000+3500)/5 = 3300
# × 1.10 for extra data from val set = 3630
LGBM_FINAL_N_ESTIMATORS = 3630

# ── Early stopping ────────────────────────────────────────────────────────────
LGBM_FIT_PARAMS = {
    "callbacks": None,   # Set in train.py: [lgb.early_stopping(200), lgb.log_evaluation(500)]
}

# ── Hyperparameter search space (Optuna) ─────────────────────────────────────
LGBM_SEARCH_SPACE = {
    "num_leaves":       ("int",   64, 1024),
    "learning_rate":    ("float", 0.001, 0.02, True),
    "subsample":        ("float", 0.6, 0.95),
    "colsample_bytree": ("float", 0.3, 0.7),
    "min_child_samples":("int",   20, 200),
    "reg_alpha":        ("float", 0.001, 0.5, True),
    "reg_lambda":       ("float", 0.001, 0.5, True),
}

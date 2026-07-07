# ── Primary configuration (validated on IEEE-CIS, ~top 5%) ──────────────────
XGB_PARAMS = {
        "n_estimators": 5000,
        "max_depth": 7,  # Perfect range for imbalanced tabular data
        "learning_rate": 0.033499048529180787,
        "subsample": 0.8832258669360352,
        "colsample_bytree": 0.3994347093531767,  # Excellent: forces feature variation
        "min_child_weight": 36,
        "reg_alpha": 0.44262501056142795,
        "reg_lambda": 0.005568074156697891,
        "scale_pos_weight": 1,  # FIXED: Removed to prevent OOF stacking distortion
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "aucpr",  # Optimizes natively for PR-AUC
        "objective": "binary:logistic",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "early_stopping_rounds": 200,
    }


# ── Final full-dataset retraining config ──────────────────────────────────────
# n_estimators = round(avg_best_iteration_across_5_folds × 1.10)
# Folds ran to max 5000 (early stopping did not trigger early).
# Using 5000 × 1.10 = 5500 to account for extra data from val set.
XGB_FINAL_N_ESTIMATORS = 5500

# ── Early stopping ────────────────────────────────────────────────────────────
XGB_FIT_PARAMS = {
    "verbose":               False,
}

# ── Hyperparameter search space (Optuna) ─────────────────────────────────────
XGB_SEARCH_SPACE = {
    "max_depth":        ("int",   6, 14),
    "learning_rate":    ("float", 0.001, 0.01, True),   # log scale
    "subsample":        ("float", 0.6, 0.9),
    "colsample_bytree": ("float", 0.3, 0.6),
    "min_child_weight": ("int",   5, 50),
    "reg_alpha":        ("float", 0.001, 0.1, True),
    "reg_lambda":       ("float", 0.001, 0.1, True),
}

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from loguru import logger

from src.models.bilstm.model import FraudBiLSTM, FocalLoss
from src.models.bilstm.dataset import BiLSTMDataset, build_sequences, SEQ_FEATURES, SEQ_LEN
from src.preprocessing.sort_and_split import expanding_window_folds


BILSTM_TRAIN_CFG = {
    "input_size":   len(SEQ_FEATURES),   # 13 (12 raw + calibrated_stacking_score)
    "hidden_size":  128,                 # 128 per direction → 256 total (bidirectional)
    "num_layers":   2,
    "dropout":      0.3,
    "epochs":       30,
    "batch_size":   1024,
    "lr":           1e-3,
    "weight_decay": 1e-5,
    "focal_gamma":  2.0,
    "focal_alpha":  0.25,
    "patience":     7,
}

# Average best epoch from CV × 1.10 (use this as fixed epoch count in full retrain mode).
# Update this after reading CV logs: avg_best_epoch × 1.10
BILSTM_FINAL_EPOCHS = 22   # default: 20 avg best epoch × 1.10 ≈ 22


def train_bilstm(
    train_df,
    val_df,
    save_dir: str = "artifacts/checkpoints",
    cfg: dict = None,
    use_cv: bool = True,
    n_folds: int = 5,
    final_epochs: Optional[int] = None,
    cv_oof_path: Optional[str] = None,
    cv_y_train_path: Optional[str] = None,
) -> dict:
    """
    Build sequences, train BiLSTM using CV, return OOF predictions and models.

    Parameters
    ----------
    use_cv          : if False, train a single model for fixed epochs (full retrain mode)
    final_epochs    : epochs when use_cv=False. Defaults to BILSTM_FINAL_EPOCHS.
    cv_oof_path     : path to bilstm_oof_preds.npy from the CV run (used to refit Platt Scaler).
    cv_y_train_path : path to y_train.npy labels matching the CV OOF rows.
    """
    if cfg is None:
        cfg = BILSTM_TRAIN_CFG

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("BiLSTM training on device: {}", device)

    # ── Build sequences ───────────────────────────────────────────────────────
    logger.info("Building training sequences…")
    X_train_full, y_train_full = build_sequences(train_df)
    logger.info("Building validation sequences…")
    X_val, y_val = build_sequences(val_df)

    np.save(save_path / "bilstm_X_train.npy", X_train_full)
    np.save(save_path / "bilstm_X_val.npy",   X_val)

    # num_workers > 0 requires multiprocessing spawn on Windows — use 0 to avoid crashes
    _n_workers   = 0 if os.name == "nt" else 4
    _pin_memory  = _n_workers > 0   # pin_memory only useful with background workers

    val_ds = BiLSTMDataset(X_val, y_val)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=_n_workers, pin_memory=_pin_memory,
    )

    oof_preds = np.zeros(len(X_train_full))
    models    = []
    cv_aucs   = []
    cv_auprs  = []

    # ── Full-dataset retrain mode ───────────────────────────────────────────────────
    if not use_cv:
        n_ep = final_epochs or BILSTM_FINAL_EPOCHS
        logger.info(
            "BiLSTM FULL RETRAIN mode — single model, {} epochs (no early stopping).", n_ep
        )

        full_ds     = BiLSTMDataset(X_train_full, y_train_full)
        full_loader = DataLoader(
            full_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=_n_workers, pin_memory=_pin_memory,
        )

        model = FraudBiLSTM(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(device)

        criterion = FocalLoss(gamma=cfg["focal_gamma"], alpha=cfg["focal_alpha"])
        optimiser = torch.optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_ep)

        for epoch in range(n_ep):
            model.train()
            train_losses = []
            for X_batch, y_batch in full_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimiser.zero_grad()
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                train_losses.append(loss.item())
            scheduler.step()
            logger.info(
                "[Full Retrain] Epoch {:3d}/{} — loss: {:.5f}",
                epoch + 1, n_ep, np.mean(train_losses),
            )

        final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(final_state, save_path / "bilstm_final.pt")
        logger.info("BiLSTM final model saved to {}", save_path / "bilstm_final.pt")

        # Val (full_train + val merged) predictions
        model.eval()
        val_probs_list = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                logits = model(X_batch.to(device))
                val_probs_list.append(torch.sigmoid(logits).cpu().numpy())
        final_val_preds = np.concatenate(val_probs_list)
        np.save(save_path / "bilstm_final_val_preds.npy", final_val_preds)

        # ── Refit Platt Scaler from CV OOF predictions ──────────────────────────
        # In full retrain mode there are no held-out OOF predictions.
        # We reuse the OOF predictions from the CV run (saved as bilstm_oof_preds.npy)
        # to refit calibration — those remain valid calibration data.
        from sklearn.linear_model import LogisticRegression as LR
        cv_oof_file   = cv_oof_path   or str(save_path / "bilstm_oof_preds.npy")
        cv_label_file = cv_y_train_path
        try:
            cv_oof = np.load(cv_oof_file)
            if cv_label_file and Path(cv_label_file).exists():
                cv_y = np.load(cv_label_file)
            else:
                # Labels for CV OOF rows = y_train_full (same ordering)
                cv_y = y_train_full
            valid_mask = cv_oof > 0
            if valid_mask.sum() > 100:
                platt_lr = LR(C=1.0, max_iter=500, solver="lbfgs")
                platt_lr.fit(cv_oof[valid_mask].reshape(-1, 1), cv_y[valid_mask])
                with open(save_path / "bilstm_platt_scaler.pkl", "wb") as fh:
                    pickle.dump(platt_lr, fh)
                logger.info(
                    "BiLSTM Platt Scaler (refitted from CV OOF) saved to {}",
                    save_path / "bilstm_platt_scaler.pkl",
                )
            else:
                logger.warning("Not enough CV OOF predictions to refit Platt Scaler.")
        except FileNotFoundError:
            logger.warning(
                "CV OOF file not found at {} — Platt Scaler not refitted.", cv_oof_file
            )

        return {
            "models":      [model],
            "oof_preds":   None,
            "val_probs":   final_val_preds,
            "oof_auc_roc": None,
            "oof_auc_pr":  None,
            "val_auc_roc": None,
            "val_auc_pr":  None,
        }

    # ── Standard CV mode ─────────────────────────────────────────────────────
    # Step 3: Track exact val indices with a boolean mask (no oof_preds > 0 heuristic)
    oof_val_mask = np.zeros(len(X_train_full), dtype=bool)
    if use_cv:
        folds = expanding_window_folds(train_df, n_folds=n_folds)
    else:
        folds = [(np.arange(len(X_train_full)), np.arange(len(X_train_full)))]

    for fold_idx, (tr_idx, vl_idx) in enumerate(folds):
        logger.info(
            "\n" + "-" * 40 + "\nBiLSTM Fold {}/{}: train={}, val={}\n" + "-" * 40,
            fold_idx + 1, len(folds), len(tr_idx), len(vl_idx),
        )

        X_tr, y_tr = X_train_full[tr_idx], y_train_full[tr_idx]
        X_vl, y_vl_fold = X_train_full[vl_idx], y_train_full[vl_idx]

        train_ds = BiLSTMDataset(X_tr, y_tr)
        fold_val_ds = BiLSTMDataset(X_vl, y_vl_fold)

        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=_n_workers, pin_memory=_pin_memory,
        )
        fold_val_loader = DataLoader(
            fold_val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
            num_workers=_n_workers, pin_memory=_pin_memory,
        )

        # ── Model + loss + optimiser ──────────────────────────────────────────────
        model = FraudBiLSTM(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(device)

        # Step 3: Use BCEWithLogitsLoss — numerically stable, no sigmoid needed in forward pass
        criterion = torch.nn.BCEWithLogitsLoss()
        optimiser = torch.optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=cfg["epochs"],
        )

        best_val_aupr  = 0.0
        patience_count = 0
        best_state     = None

        for epoch in range(cfg["epochs"]):
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimiser.zero_grad()
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                train_losses.append(loss.item())

            model.eval()
            val_probs = []
            with torch.no_grad():
                for X_batch, _ in fold_val_loader:
                    logits = model(X_batch.to(device))
                    probs  = torch.sigmoid(logits).cpu().numpy()
                    val_probs.append(probs)

            val_probs = np.concatenate(val_probs)
            val_aupr  = average_precision_score(y_vl_fold, val_probs)
            val_auc   = roc_auc_score(y_vl_fold, val_probs)
            scheduler.step()

            logger.info(
                "Epoch {:3d}/{} — loss: {:.5f}  val AUC-ROC: {:.5f}  AUC-PR: {:.5f}",
                epoch + 1, cfg["epochs"], np.mean(train_losses), val_auc, val_aupr,
            )

            if val_aupr > best_val_aupr:
                best_val_aupr  = val_aupr
                patience_count = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= cfg["patience"]:
                    logger.info("Early stopping at epoch {}", epoch + 1)
                    break

        model.load_state_dict(best_state)
        torch.save(best_state, save_path / f"bilstm_fold{fold_idx}.pt")
        models.append(model)

        # ── OOF Predictions for this fold ────────────────────────────────────────
        model.eval()
        fold_probs_final = []
        with torch.no_grad():
            for X_batch, _ in fold_val_loader:
                logits = model(X_batch.to(device))
                fold_probs_final.append(torch.sigmoid(logits).cpu().numpy())
        fold_probs_final = np.concatenate(fold_probs_final)
        oof_preds[vl_idx] = fold_probs_final
        oof_val_mask[vl_idx] = True   # Step 3: track which rows have OOF predictions

        cv_aucs.append(roc_auc_score(y_vl_fold, fold_probs_final))
        cv_auprs.append(average_precision_score(y_vl_fold, fold_probs_final))

    # Step 3: Compute true, uncalibrated OOF metrics using the exact val mask
    val_indices = oof_val_mask
    oof_auc  = roc_auc_score(y_train_full[val_indices], oof_preds[val_indices])
    oof_aupr = average_precision_score(y_train_full[val_indices], oof_preds[val_indices])
    logger.info("BiLSTM Raw OOF (uncalibrated) — AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}", oof_auc, oof_aupr)

    # ── Final test/val predictions (average across K models) ───────────────────
    val_preds = None
    if val_df is not None:
        all_fold_preds = []
        for model in models:
            model.eval()
            v_probs = []
            with torch.no_grad():
                for X_batch, _ in val_loader:
                    logits = model(X_batch.to(device))
                    v_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_fold_preds.append(np.concatenate(v_probs))
            
        val_preds = np.mean(all_fold_preds, axis=0)
        val_auc  = roc_auc_score(y_val, val_preds)
        val_aupr = average_precision_score(y_val, val_preds)
        logger.info("BiLSTM Val (Ensemble) — AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}", val_auc, val_aupr)

    np.save(save_path / "bilstm_oof_preds.npy", oof_preds)
    np.save(save_path / "bilstm_val_probs.npy", val_preds)

    # ── Step 4: Global Isotonic Calibration (applied AFTER the full CV loop) ─────────
    # This is the ONLY place calibration is applied in the entire BiLSTM pipeline.
    # It uses the completely aggregated, uncalibrated OOF array to fit a single
    # global scaler. This asset is used strictly to calibrate final submission predictions.
    if val_indices.sum() > 100:
        global_calibrator = CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
            method="isotonic",
            cv=5,
        )
        global_calibrator.fit(oof_preds[val_indices].reshape(-1, 1), y_train_full[val_indices])
        with open(save_path / "bilstm_global_calibrator.pkl", "wb") as fh:
            pickle.dump(global_calibrator, fh)
        logger.info("BiLSTM Global Calibrator saved to {}", save_path / "bilstm_global_calibrator.pkl")

        # Apply to val/test preds for final submission use only
        if val_preds is not None:
            val_preds_cal = global_calibrator.predict_proba(val_preds.reshape(-1, 1))[:, 1]
            np.save(save_path / "bilstm_val_probs_calibrated.npy", val_preds_cal)
            cal_auc  = roc_auc_score(y_val, val_preds_cal)
            cal_aupr = average_precision_score(y_val, val_preds_cal)
            logger.info("BiLSTM Calibrated Val — AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}", cal_auc, cal_aupr)
    else:
        logger.warning("Not enough OOF predictions to fit Global Calibrator (n={}).", val_indices.sum())

    return {
        "models":      models,
        "oof_preds":   oof_preds,
        "val_probs":   val_preds,
        "oof_auc_roc": oof_auc,
        "oof_auc_pr":  oof_aupr,
        "val_auc_roc": val_auc if val_preds is not None else None,
        "val_auc_pr":  val_aupr if val_preds is not None else None,
    }


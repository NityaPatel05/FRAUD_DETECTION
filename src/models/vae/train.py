from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from loguru import logger

from src.models.vae.model import FraudVAE
from src.models.vae.dataset import VAEDataset


# ── Training config ───────────────────────────────────────────────────────────
VAE_TRAIN_CFG = {
    "latent_dim":   16,
    "beta":         1.0,
    "epochs":       50,
    "batch_size":   2048,
    "lr":           1e-3,
    "weight_decay": 1e-5,
    "patience":     10,         # early stopping on val recon loss
}


def train_vae(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_dir: str = "artifacts/checkpoints",
    cfg: dict = None,
) -> dict:
    """
    Train VAE and compute anomaly scores for train + val sets.

    Parameters
    ----------
    X_train, y_train : full training features + labels (legit filtered internally)
    X_val,   y_val   : validation features + labels (both legit + fraud for scoring)

    Returns
    -------
    dict with 'model', 'train_scores', 'val_scores', 'score_min', 'score_max'
    """
    if cfg is None:
        cfg = VAE_TRAIN_CFG

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim  = X_train.shape[1]
    latent_dim = cfg["latent_dim"]
    beta       = cfg["beta"]

    logger.info(
        "VAE training: input_dim={}, latent_dim={}, device={}, "
        "legit train samples={} / {} total",
        input_dim, latent_dim, device,
        int((y_train == 0).sum()), len(y_train),
    )

    from sklearn.preprocessing import StandardScaler
    import pickle

    # Scale features using StandardScaler fit on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # Save fitted scaler for inference
    with open(save_path / "vae_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved VAE scaler to {}", save_path / "vae_scaler.pkl")

    # ── Datasets ──────────────────────────────────────────────────────────────
    # train_ds   : legit only   — VAE learns to reconstruct normal transactions
    # val_es_ds  : legit only   — used for early stopping / LR scheduler (in-distribution)
    # val_full_ds: legit+fraud  — used ONLY for final AUC evaluation
    train_ds    = VAEDataset(X_train_scaled, y_train, legit_only=True)
    val_es_ds   = VAEDataset(X_val_scaled,   y_val,   legit_only=True)   # in-distribution val
    val_full_ds = VAEDataset(X_val_scaled,   y_val,   legit_only=False)  # full val for AUC

    # num_workers > 0 requires multiprocessing spawn on Windows which fails in Jupyter — use 0
    import os as _os
    _n_workers = 0 if _os.name == "nt" else 0
    _pin_memory = _n_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=_n_workers, pin_memory=_pin_memory, drop_last=True,
    )
    # Early-stopping loader: legit only — gives a stable, finite reconstruction loss
    val_es_loader = DataLoader(
        val_es_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=_n_workers, pin_memory=_pin_memory,
    )
    # Full val loader for final AUC scoring only (not used in training loop)
    val_full_loader = DataLoader(
        val_full_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=_n_workers, pin_memory=_pin_memory,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FraudVAE(input_dim=input_dim, latent_dim=latent_dim, beta=beta).to(device)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(cfg["epochs"]):
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            optimiser.zero_grad()
            x_hat, mu, logvar = model(x)
            loss, recon, kl   = model.loss(x, x_hat, mu, logvar)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_losses.append(loss.item())

        # ── Validation (legit-only, for early stopping) ───────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_es_loader:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                x_hat, mu, logvar = model(x)
                # Full VAE loss on in-distribution legit data — finite and meaningful
                loss, _, _        = model.loss(x, x_hat, mu, logvar)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)
        scheduler.step(avg_val)

        if epoch % 5 == 0 or epoch < 5:
            logger.info(
                "Epoch {:3d}/{} — train_loss: {:.5f}  val_loss: {:.5f}",
                epoch + 1, cfg["epochs"], avg_train, avg_val,
            )

        # ── Early stopping ────────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss   = avg_val
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                logger.info("Early stopping at epoch {}", epoch + 1)
                break

    # ── Load best weights ─────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    torch.save(best_state, save_path / "vae_best.pt")
    logger.info("VAE best model saved to {}", save_path / "vae_best.pt")

    # ── Compute anomaly scores for ALL rows (train legit + fraud) ─────────────
    model.eval()

    def compute_scores(X_arr: np.ndarray) -> np.ndarray:
        scores = []
        ds     = VAEDataset(X_arr, legit_only=False)
        loader = DataLoader(ds, batch_size=4096, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                x = batch.to(device) if not isinstance(batch, (list, tuple)) else batch[0].to(device)
                s = model.anomaly_score(x)
                scores.append(s.cpu().numpy())
        return np.concatenate(scores)

    raw_train_scores = compute_scores(X_train_scaled)
    raw_val_scores   = compute_scores(X_val_scaled)

    # Normalise using train distribution (min-max on legit train rows only)
    legit_mask   = (y_train == 0)
    score_min    = raw_train_scores[legit_mask].min()
    score_max    = raw_train_scores[legit_mask].max() + 1e-8

    def normalise(s):
        return np.clip((s - score_min) / (score_max - score_min), 0.0, 1.0)

    train_scores = normalise(raw_train_scores)
    val_scores   = normalise(raw_val_scores)

    # Evaluate anomaly detection quality
    val_auc  = roc_auc_score(y_val, val_scores)
    val_aupr = average_precision_score(y_val, val_scores)
    logger.info(
        "VAE anomaly score — Val AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}",
        val_auc, val_aupr,
    )

    np.save(save_path / "vae_train_scores.npy", train_scores)
    np.save(save_path / "vae_val_scores.npy",   val_scores)
    np.save(save_path / "vae_score_stats.npy",  np.array([score_min, score_max]))

    return {
        "model":         model,
        "train_scores":  train_scores,
        "val_scores":    val_scores,
        "score_min":     score_min,
        "score_max":     score_max,
        "val_auc_roc":   val_auc,
        "val_auc_pr":    val_aupr,
    }

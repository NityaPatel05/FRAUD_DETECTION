from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from loguru import logger

from src.models.gnn.model import HeteroSAGE
from src.models.gnn.graph_builder import build_graph_data
from src.preprocessing.sort_and_split import expanding_window_folds

try:
    from torch_geometric.loader import NeighborLoader
except ImportError:
    raise ImportError(
        "PyTorch Geometric not installed. "
        "Run: pip install torch-geometric"
    )


GNN_TRAIN_CFG = {
    "proj_dim":      128,
    "embed_dim":     64,
    "epochs":        30,
    "lr":            1e-3,
    "weight_decay":  1e-5,
    "fraud_weight":  20.0,    # cost-sensitive: class weight for fraud
    "patience":      7,
    "batch_size":    1024,    # ← mini-batch size for NeighborLoader
    "num_neighbors": [15, 10],# ← 2-hop neighbourhood samples
}


def _build_neighbor_loader(
    data,
    target_node_type: str,
    input_nodes,
    batch_size: int,
    num_neighbors: list,
    shuffle: bool = False,
) -> "NeighborLoader":
    """Helper to construct a NeighborLoader for a given set of node IDs."""
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=(target_node_type, input_nodes),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,      # 0 = main process; avoids Windows multiprocessing issues
    )


def _model_from_data(data, cfg: dict, device: torch.device) -> HeteroSAGE:
    """Instantiate a HeteroSAGE from a HeteroData object's metadata."""
    in_dims = {
        ntype: data[ntype].x.size(1)
        for ntype in data.node_types
    }
    model = HeteroSAGE(
        in_dims=in_dims,
        proj_dim=cfg["proj_dim"],
        embed_dim=cfg["embed_dim"],
        metadata=data.metadata(),
    ).to(device)
    return model


@torch.no_grad()
def _predict_full(model: HeteroSAGE, data, device: torch.device) -> np.ndarray:
    """
    Run a full-graph inference pass to get fraud probability per transaction.
    This is only used OFFLINE (after training) for OOF eval and embedding export.
    It is NOT called during training epochs — training uses NeighborLoader.
    """
    model.eval()
    data_d = data.to(device)
    logits = model(data_d.x_dict, data_d.edge_index_dict)
    probs  = torch.sigmoid(logits).cpu().numpy()
    return probs


@torch.no_grad()
def _get_all_embeddings(
    model: HeteroSAGE,
    data,
    device: torch.device,
) -> dict:
    """Full-graph forward pass to extract all node embeddings."""
    model.eval()
    data_d = data.to(device)
    return {
        k: v.cpu().numpy()
        for k, v in model.get_embeddings(data_d.x_dict, data_d.edge_index_dict).items()
    }


def train_gnn(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    top_v_cols: list = None,
    save_dir: str = "artifacts/checkpoints",
    cfg: dict = None,
    use_cv: bool = True,
    n_folds: int = 5,
) -> dict:
    """
    Train HeteroSAGE with NeighborLoader mini-batching and expanding CV.

    Parameters
    ----------
    train_df  : Feature-engineered training DataFrame (sorted by TransactionDT).
    val_df    : Optional held-out validation DataFrame for final evaluation.
    top_v_cols: Top V columns by SHAP to add as extra transaction node features.
    save_dir  : Directory to save model checkpoints and embeddings.
    cfg       : Training hyperparameters (uses GNN_TRAIN_CFG if None).
    use_cv    : Whether to use expanding-window CV (recommended).
    n_folds   : Number of CV folds.

    Returns
    -------
    dict with keys:
      oof_preds, val_probs, ring_risk_score, card_embeddings, card_map,
      oof_auc_roc, oof_auc_pr, val_auc_roc, val_auc_pr, models
    """
    if cfg is None:
        cfg = GNN_TRAIN_CFG

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("GNN training on device: {}", device)

    oof_preds = np.zeros(len(train_df))
    models    = []
    cv_aucs   = []
    cv_auprs  = []

    if use_cv:
        folds = expanding_window_folds(train_df, n_folds=n_folds)
    else:
        folds = [(np.arange(len(train_df)), np.arange(len(train_df)))]

    # ── CV Loop ────────────────────────────────────────────────────────────────
    for fold_idx, (tr_idx, vl_idx) in enumerate(folds):
        logger.info(
            "\n{}\nGNN Fold {}/{}: train={}, val={}\n{}",
            "-" * 50, fold_idx + 1, len(folds), len(tr_idx), len(vl_idx), "-" * 50,
        )

        # Build graph on fold data (no future leakage: only rows up to max_idx)
        max_idx = len(tr_idx) + len(vl_idx)
        fold_df = train_df.iloc[:max_idx].reset_index(drop=True)

        logger.info("Building PyG HeteroData for Fold {}…", fold_idx + 1)
        data = build_graph_data(fold_df, top_v_cols=top_v_cols)

        # Node labels live only on transaction nodes
        y_fold = fold_df["isFraud"].values.astype(np.float32)

        # Train / val node indices within this fold's graph
        tr_node_ids = torch.arange(len(tr_idx), dtype=torch.long)
        vl_node_ids = torch.arange(len(tr_idx), len(tr_idx) + len(vl_idx), dtype=torch.long)
        y_vl        = y_fold[len(tr_idx):]

        # ── Build NeighborLoaders ─────────────────────────────────────────────
        train_loader = _build_neighbor_loader(
            data, "transaction", tr_node_ids,
            batch_size=cfg["batch_size"], num_neighbors=cfg["num_neighbors"], shuffle=True,
        )
        val_loader = _build_neighbor_loader(
            data, "transaction", vl_node_ids,
            batch_size=cfg["batch_size"] * 2, num_neighbors=cfg["num_neighbors"], shuffle=False,
        )

        # ── Model, loss, optimiser ────────────────────────────────────────────
        model = _model_from_data(data, cfg, device)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([cfg["fraud_weight"]], device=device)
        )
        optimiser = torch.optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", factor=0.5, patience=3,
        )

        best_val_aupr  = 0.0
        patience_count = 0
        best_state     = None

        for epoch in range(cfg["epochs"]):
            # ── Training ──────────────────────────────────────────────────────
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimiser.zero_grad()

                logits = model(batch.x_dict, batch.edge_index_dict)

                # batch["transaction"].y has labels for the batch transaction nodes
                # NeighborLoader places "seed" (input) nodes first in the batch
                n_seed = batch["transaction"].batch_size
                seed_logits = logits[:n_seed]
                seed_labels = batch["transaction"].y[:n_seed].float()

                loss = criterion(seed_logits, seed_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(train_loader), 1)

            # ── Validation ────────────────────────────────────────────────────
            model.eval()
            val_probs_ep = []
            val_labels_ep = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    logits = model(batch.x_dict, batch.edge_index_dict)
                    n_seed = batch["transaction"].batch_size
                    probs  = torch.sigmoid(logits[:n_seed]).cpu().numpy()
                    labs   = batch["transaction"].y[:n_seed].cpu().numpy()
                    val_probs_ep.append(probs)
                    val_labels_ep.append(labs)

            val_probs_ep  = np.concatenate(val_probs_ep)
            val_labels_ep = np.concatenate(val_labels_ep)

            val_aupr = average_precision_score(val_labels_ep, val_probs_ep)
            val_auc  = roc_auc_score(val_labels_ep, val_probs_ep)
            scheduler.step(val_aupr)

            logger.info(
                "Epoch {:3d}/{} — loss: {:.5f}  val AUC-ROC: {:.5f}  AUC-PR: {:.5f}",
                epoch + 1, cfg["epochs"], avg_loss, val_auc, val_aupr,
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

        # Load best weights
        model.load_state_dict(best_state)
        torch.save(best_state, save_path / f"gnn_fold{fold_idx}.pt")
        models.append(model.cpu())  # store on CPU to save VRAM

        # ── OOF predictions (full-graph pass, offline) ────────────────────────
        all_probs = _predict_full(model.to(device), data, device)
        oof_preds[vl_idx] = all_probs[len(tr_idx):]

        cv_aucs.append(roc_auc_score(y_vl, oof_preds[vl_idx]))
        cv_auprs.append(average_precision_score(y_vl, oof_preds[vl_idx]))
        model.cpu()  # free VRAM after fold

    # Overall OOF metrics
    valid_mask = oof_preds > 0
    oof_auc  = roc_auc_score(train_df["isFraud"].values[valid_mask], oof_preds[valid_mask])
    oof_aupr = average_precision_score(train_df["isFraud"].values[valid_mask], oof_preds[valid_mask])
    logger.info("GNN OOF — AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}", oof_auc, oof_aupr)

    np.save(save_path / "gnn_oof_preds.npy", oof_preds)

    # ── Validation set + card embeddings export ────────────────────────────────
    val_preds     = None
    ring_risk     = None
    card_embs_out = None
    card_map_out  = None

    # ── Final Eval & Embeddings Export ─────────────────────────────────────────
    logger.info("Building final graph for evaluation and embedding export…")
    eval_df   = pd.concat([train_df, val_df]).reset_index(drop=True) if val_df is not None else train_df
    eval_data = build_graph_data(eval_df, top_v_cols=top_v_cols)

    # Export card embeddings
    best_model = models[-1].to(device)
    all_embs   = _get_all_embeddings(best_model, eval_data, device)
    card_embs_out = all_embs.get("card")
    card_map_out  = eval_data.card_map

    np.save(save_path / "gnn_card_embeddings.npy", card_embs_out)
    with open(save_path / "gnn_card_map.pkl", "wb") as f:
        pickle.dump(card_map_out, f)

    if val_df is not None:
        n_tr = len(train_df)
        all_fold_preds = []
        for m in models:
            m.to(device)
            probs = _predict_full(m, eval_data, device)
            all_fold_preds.append(probs[n_tr:])  # only val rows
            m.cpu()

        val_preds = np.mean(all_fold_preds, axis=0)
        ring_risk = val_preds.copy()

        y_val_true = val_df["isFraud"].values
        val_auc  = roc_auc_score(y_val_true, val_preds)
        val_aupr = average_precision_score(y_val_true, val_preds)
        logger.info("GNN Val — AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}", val_auc, val_aupr)

        np.save(save_path / "gnn_val_probs.npy", val_preds)
        np.save(save_path / "gnn_ring_risk.npy", ring_risk)
        
    best_model.cpu()

    return {
        "models":          models,
        "oof_preds":       oof_preds,
        "val_probs":       val_preds,
        "ring_risk_score": ring_risk,
        "card_embeddings": card_embs_out,
        "card_map":        card_map_out,
        "oof_auc_roc":     oof_auc,
        "oof_auc_pr":      oof_aupr,
        "val_auc_roc":     roc_auc_score(val_df["isFraud"].values, val_preds) if val_preds is not None else None,
        "val_auc_pr":      average_precision_score(val_df["isFraud"].values, val_preds) if val_preds is not None else None,
    }

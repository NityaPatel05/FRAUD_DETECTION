from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from loguru import logger

try:
    from torch_geometric.data import HeteroData
except ImportError:
    raise ImportError(
        "PyTorch Geometric not installed. "
        "Run: pip install torch-geometric"
    )


# ── Node feature definitions ───────────────────────────────────────────────────
TRANSACTION_NODE_FEATS = [
    "TransactionAmt_log", "amt_zscore",
    "hour_sin", "hour_cos", "is_late_night",
    "ProductCD_encoded",
]

CARD_NODE_FEATS = [
    "card1_freq", "card1_amt_mean", "card1_amt_std", "card1_tx_count",
    "C1_uid_mean", "D1_norm_uid_mean", "M_match_count",
]

ADDRESS_NODE_FEATS = [
    "addr1_freq", "addr1_tx_count", "addr1_amt_mean",
]

EMAIL_NODE_FEATS = [
    "P_emaildomain_freq", "email_trusted", "P_email_is_free",
]

DEVICE_NODE_FEATS = [
    "device_os_encoded", "device_browser_encoded",
    "DeviceInfo_freq", "device_mobile",
]

# Edge features added to each transaction→entity edge
EDGE_FEATS = [
    "TransactionAmt_log",
    "hour_sin",
    "hour_cos",
    "ProductCD_encoded",
]

DECAY_LAMBDA = 0.0001


def _safe_get(df: pd.DataFrame, cols: list, fill: float = 0.0) -> np.ndarray:
    """Return float32 array for `cols`, zeroing any missing columns."""
    full = np.zeros((len(df), len(cols)), dtype=np.float32)
    for j, col in enumerate(cols):
        if col in df.columns:
            full[:, j] = df[col].fillna(fill).values.astype(np.float32)
        else:
            logger.debug("GNN feat missing (zeroed): {}", col)
    return full


def build_graph_data(
    df: pd.DataFrame,
    top_v_cols: list = None,
    decay_lambda: float = DECAY_LAMBDA,
) -> "HeteroData":
    """
    Convert a feature-engineered DataFrame into a PyG HeteroData object.

    The result is directly consumed by PyG's NeighborLoader for scalable
    mini-batch training.

    Parameters
    ----------
    df           : Feature-engineered DataFrame sorted by TransactionDT.
    top_v_cols   : Optional extra V-columns appended to transaction node feats.
    decay_lambda : Coefficient for temporal edge-weight decay.

    Returns
    -------
    data : torch_geometric.data.HeteroData
           Contains x and y for each node type, plus edge_index and edge_attr.
    """
    n = len(df)
    logger.info("Building PyG HeteroData: {} transactions…", n)

    # ── Node ID maps ────────────────────────────────────────────────────────
    def build_id_map(series: pd.Series) -> dict:
        unique_vals = series.fillna("unknown").astype(str).unique()
        return {v: i for i, v in enumerate(unique_vals)}

    card_map   = build_id_map(df["card1"])
    addr_map   = build_id_map(df["addr1"])
    email_map  = build_id_map(df["P_emaildomain"])
    device_map = build_id_map(df["DeviceInfo"])

    # ── Transaction node features ────────────────────────────────────────────
    txn_feat_cols = TRANSACTION_NODE_FEATS.copy()
    if top_v_cols:
        txn_feat_cols += [c for c in top_v_cols if c in df.columns]
    txn_feats = _safe_get(df, txn_feat_cols)

    # ── Entity node features (mean-aggregated per entity) ────────────────────
    def agg_entity_feats(group_col: str, feat_cols: list, id_map: dict) -> np.ndarray:
        """Mean-pool transaction features per entity node."""
        safe_cols = [c for c in feat_cols if c in df.columns]
        node_df = (
            df.assign(_gid=df[group_col].fillna("unknown").astype(str))
              .groupby("_gid")[safe_cols]
              .mean()
              .reindex(list(id_map.keys()))
              .fillna(0.0)
        )
        # Rebuild full matrix with zeros for missing columns
        full = np.zeros((len(id_map), len(feat_cols)), dtype=np.float32)
        for j, col in enumerate(feat_cols):
            if col in node_df.columns:
                full[:, j] = node_df[col].values.astype(np.float32)
        return full

    card_feats   = agg_entity_feats("card1",         CARD_NODE_FEATS,    card_map)
    addr_feats   = agg_entity_feats("addr1",         ADDRESS_NODE_FEATS, addr_map)
    email_feats  = agg_entity_feats("P_emaildomain", EMAIL_NODE_FEATS,   email_map)
    device_feats = agg_entity_feats("DeviceInfo",    DEVICE_NODE_FEATS,  device_map)

    # ── Edge indices: transaction → entity ───────────────────────────────────
    txn_ids     = np.arange(n, dtype=np.int64)
    card_ids    = np.array([card_map[v]   for v in df["card1"].fillna("unknown").astype(str)],  dtype=np.int64)
    addr_ids    = np.array([addr_map[v]   for v in df["addr1"].fillna("unknown").astype(str)],  dtype=np.int64)
    email_ids   = np.array([email_map[v]  for v in df["P_emaildomain"].fillna("unknown").astype(str)], dtype=np.int64)
    device_ids  = np.array([device_map[v] for v in df["DeviceInfo"].fillna("unknown").astype(str)],    dtype=np.int64)

    # ── Edge attributes ───────────────────────────────────────────────────────
    edge_base   = _safe_get(df, EDGE_FEATS)  # (n, 4)
    gap         = df["uid_gap_last"].fillna(86400).clip(lower=0).values.astype(np.float32)
    decay       = np.exp(-decay_lambda * gap).reshape(-1, 1)           # (n, 1)
    edge_attr   = np.concatenate([edge_base, decay], axis=1)           # (n, 5)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

    # ── Target labels ─────────────────────────────────────────────────────────
    y = torch.tensor(
        df["isFraud"].values.astype(np.int64) if "isFraud" in df.columns
        else np.zeros(n, dtype=np.int64),
    )

    # ── Assemble HeteroData ───────────────────────────────────────────────────
    data = HeteroData()

    # Node features
    data["transaction"].x = torch.tensor(txn_feats,   dtype=torch.float32)
    data["card"].x        = torch.tensor(card_feats,  dtype=torch.float32)
    data["addr"].x        = torch.tensor(addr_feats,  dtype=torch.float32)
    data["email"].x       = torch.tensor(email_feats, dtype=torch.float32)
    data["device"].x      = torch.tensor(device_feats, dtype=torch.float32)

    # Labels on transaction nodes
    data["transaction"].y = y

    # Forward edges (transaction → entity) with temporal edge attrs
    fwd  = torch.from_numpy(np.stack([txn_ids, card_ids],   axis=0))
    data["transaction", "uses_card",   "card"].edge_index   = fwd
    data["transaction", "uses_card",   "card"].edge_attr    = edge_attr_t

    fwd  = torch.from_numpy(np.stack([txn_ids, addr_ids],   axis=0))
    data["transaction", "from_addr",   "addr"].edge_index   = fwd
    data["transaction", "from_addr",   "addr"].edge_attr    = edge_attr_t

    fwd  = torch.from_numpy(np.stack([txn_ids, email_ids],  axis=0))
    data["transaction", "uses_email",  "email"].edge_index  = fwd
    data["transaction", "uses_email",  "email"].edge_attr   = edge_attr_t

    fwd  = torch.from_numpy(np.stack([txn_ids, device_ids], axis=0))
    data["transaction", "uses_device", "device"].edge_index = fwd
    data["transaction", "uses_device", "device"].edge_attr  = edge_attr_t

    # Reverse edges (entity → transaction) — required by SAGEConv for
    # information to flow back to transaction nodes during aggregation
    data["card",   "rev_uses_card",   "transaction"].edge_index = torch.from_numpy(np.stack([card_ids,   txn_ids], axis=0))
    data["addr",   "rev_from_addr",   "transaction"].edge_index = torch.from_numpy(np.stack([addr_ids,   txn_ids], axis=0))
    data["email",  "rev_uses_email",  "transaction"].edge_index = torch.from_numpy(np.stack([email_ids,  txn_ids], axis=0))
    data["device", "rev_uses_device", "transaction"].edge_index = torch.from_numpy(np.stack([device_ids, txn_ids], axis=0))

    # Store maps for inference-time node ID resolution
    data.card_map   = card_map
    data.addr_map   = addr_map
    data.email_map  = email_map
    data.device_map = device_map

    logger.info(
        "HeteroData built: {} txn, {} card, {} addr, {} email, {} device nodes.",
        n, len(card_map), len(addr_map), len(email_map), len(device_map),
    )
    return data

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, to_hetero
    from torch_geometric.data import HeteroData
except ImportError:
    raise ImportError(
        "PyTorch Geometric not installed. "
        "Run: pip install torch-geometric"
    )


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


# ── Input projection: each node type has different feature dimension ───────────
class HeteroInputProjection(nn.Module):
    """Project each node type's raw features to a common embedding dimension."""

    def __init__(self, in_dims: Dict[str, int], out_dim: int = 128):
        super().__init__()
        self.projections = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(in_dim, out_dim),
                Mish(),
                nn.LayerNorm(out_dim),
            )
            for ntype, in_dim in in_dims.items()
        })

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            ntype: self.projections[ntype](x)
            for ntype, x in x_dict.items()
            if ntype in self.projections
        }


# ── Homogeneous base model that to_hetero() will convert ──────────────────────
class HomoSAGE(nn.Module):
    """
    A simple 2-layer GraphSAGE applied to a homogeneous graph.
    PyG's `to_hetero()` will replicate this for each edge type in HeteroData.
    """
    def __init__(self, in_channels: int, hidden: int = 128, out_channels: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels,  hidden,      aggr="mean")
        self.conv2 = SAGEConv(hidden,       out_channels, aggr="mean")
        self.act   = Mish()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(out_channels)
        self.drop  = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.norm1(self.conv1(x, edge_index))))
        x = self.norm2(self.conv2(x, edge_index))
        return x


# ── Main HeteroSAGE model ──────────────────────────────────────────────────────
class HeteroSAGE(nn.Module):
    """
    Full Heterogeneous GraphSAGE for fraud detection.

    1. HeteroInputProjection: per-node-type linear → 128-dim
    2. to_hetero(HomoSAGE):   message passing across all edge types → 64-dim
    3. MLP classifier on transaction node embeddings → fraud logit

    Input dim (after projection): 128 for all node types.
    Output: per-transaction fraud logit scalar.
    """

    def __init__(
        self,
        in_dims:     Dict[str, int],    # {node_type: raw_feature_dim}
        proj_dim:    int = 128,          # common projection dim
        embed_dim:   int = 64,           # final embedding dim after SAGE layers
        metadata:    tuple = None,       # HeteroData.metadata() — passed by train.py
    ):
        super().__init__()

        self.projection = HeteroInputProjection(in_dims, out_dim=proj_dim)

        # Build homogeneous SAGE and convert to heterogeneous
        base_sage = HomoSAGE(
            in_channels=proj_dim,
            hidden=proj_dim,
            out_channels=embed_dim,
        )
        self.sage = to_hetero(base_sage, metadata=metadata, aggr="mean")

        # MLP classifier on transaction node embeddings
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            Mish(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_dict:          Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_dict          : {node_type: feature_tensor}   (from HeteroData / mini-batch)
        edge_index_dict : {edge_type_tuple: edge_index} (from HeteroData / mini-batch)

        Returns
        -------
        logits : (n_transaction_nodes,) — raw fraud logits (before sigmoid)
        """
        # Project each node type to common dim
        h_dict = self.projection(x_dict)

        # Run HeteroSAGE message passing
        h_dict = self.sage(h_dict, edge_index_dict)

        # Classify transaction nodes
        txn_emb = h_dict["transaction"]               # (n_txn, embed_dim)
        logits  = self.classifier(txn_emb).squeeze(-1)  # (n_txn,)
        return logits

    def get_embeddings(
        self,
        x_dict:          Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Return all node-type embeddings (64-dim each).
        Used for:
          - Saving card embeddings to disk/Redis for tree model injection.
          - Ring detection (device/email node embeddings).
        """
        h_dict = self.projection(x_dict)
        h_dict = self.sage(h_dict, edge_index_dict)
        return h_dict

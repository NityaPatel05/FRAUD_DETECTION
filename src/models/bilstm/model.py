from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class SelfAttention(nn.Module):
    """
    Soft self-attention over sequence dimension.
    Learns which timesteps are most informative for fraud prediction.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden : (batch, seq_len, hidden_dim)

        Returns
        -------
        context : (batch, hidden_dim) — weighted sum of hidden states
        """
        scores  = self.attn(hidden).squeeze(-1)       # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)            # (batch, seq_len)
        context = torch.bmm(
            weights.unsqueeze(1), hidden
        ).squeeze(1)                                   # (batch, hidden_dim)
        return context


class FraudBiLSTM(nn.Module):
    """
    2-layer Bidirectional LSTM with self-attention for fraud detection.
    Input: (batch, seq_len=10, n_features=13)
    Output: (batch, 1) fraud probability
    """

    def __init__(
        self,
        input_size:  int = 13,    # 12 raw features + calibrated_stacking_score
        hidden_size: int = 128,   # 128 per direction → 256 total output (bidirectional)
        num_layers:  int = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Bidirectional LSTM → output dim = hidden_size * 2
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2  # hidden_size=128 per direction → 256 total concat output

        self.dropout1  = nn.Dropout(dropout)
        self.attention = SelfAttention(lstm_out_dim)
        self.dropout2  = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            Mish(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        Returns: (batch,) logits
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden*2)
        lstm_out    = self.dropout1(lstm_out)
        context     = self.attention(lstm_out)  # (batch, hidden*2)
        context     = self.dropout2(context)
        logits      = self.classifier(context).squeeze(-1)   # (batch,)
        return logits


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Handles class imbalance without SMOTE.
    γ=2, α=0.25 per the spec (matching RetinaNet defaults for fraud detection).

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (batch,) raw logits
        targets : (batch,) float 0/1
        """
        probs   = torch.sigmoid(logits)
        pt      = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(pt, self.alpha),
                              torch.full_like(pt, 1 - self.alpha))
        focal_w = (1 - pt) ** self.gamma
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss    = alpha_t * focal_w * bce
        return loss.mean()

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    """Dataset that wraps a numpy feature array."""

    def __init__(self, X: np.ndarray, y: np.ndarray = None, legit_only: bool = False):
        """
        Parameters
        ----------
        X           : feature array (already median-imputed, shape [N, F])
        y           : labels (0/1). If legit_only=True, only rows where y==0 are kept.
        legit_only  : if True, filter to non-fraud rows only (for VAE training)
        """
        if legit_only and y is not None:
            mask = (y == 0)
            X = X[mask]
            y = y[mask] if y is not None else None

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

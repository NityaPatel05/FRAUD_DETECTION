from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x)).  Better gradient flow than ReLU."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 80),
            Mish(),
            nn.BatchNorm1d(80),
            nn.Linear(80, 40),
            Mish(),
            nn.BatchNorm1d(40),
        )
        self.mu_layer    = nn.Linear(40, latent_dim)
        self.logvar_layer = nn.Linear(40, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu_layer(h), self.logvar_layer(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 40),
            Mish(),
            nn.BatchNorm1d(40),
            nn.Linear(40, 80),
            Mish(),
            nn.BatchNorm1d(80),
            nn.Linear(80, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FraudVAE(nn.Module):
    """
    VAE trained on legitimate transactions only.
    High reconstruction error ≈ anomalous (potentially fraudulent) transaction.
    """

    def __init__(self, input_dim: int, latent_dim: int = 16, beta: float = 1.0):
        super().__init__()
        self.encoder   = VAEEncoder(input_dim, latent_dim)
        self.decoder   = VAEDecoder(latent_dim, input_dim)
        self.latent_dim = latent_dim
        self.beta       = beta          # β-VAE weight on KL term

    def reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterisation trick: z = μ + ε·σ  (ε ~ N(0,1))."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z          = self.reparameterise(mu, logvar)
        x_hat      = self.decoder(z)
        return x_hat, mu, logvar

    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Total VAE loss = reconstruction MSE + β * KL divergence.

        Returns (total_loss, recon_loss, kl_loss).
        """
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        # Clamp logvar to prevent exp() overflow in float32 (exp(88.7) ≈ 3.4e38 = max float32).
        # This is standard VAE practice — logvar outside [-10, 10] is numerically meaningless.
        logvar_c = logvar.clamp(-10, 10)

        # KL divergence: -0.5 * Σ(1 + logvar - μ² - e^logvar)
        kl_loss = -0.5 * torch.mean(
            1 + logvar_c - mu.pow(2) - logvar_c.exp()
        )

        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample reconstruction error.
        Returns raw MSE per sample (not normalised — normalise after batch inference).
        """
        self.eval()
        x_hat, _, _ = self.forward(x)
        # Per-sample MSE (mean over feature dim)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=1)

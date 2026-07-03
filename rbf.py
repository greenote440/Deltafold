"""
Gaussian radial-basis distance encoding (plan_implementation §1, Modification 1).

Replaces the Transformer-style *sinusoidal* encoding of the Cα–Cα edge distance,
which is ill-suited to the physical contact scale: with base 10000 over 16 dims
on distances of 0–~20 Å, the high-index bands have ω_k ≈ 3e-4, so sin(d·ω_k)≈0
and cos≈1 for every contact distance — roughly half the dimensions are ~constant
(dead capacity, plan §1).

The Gaussian RBF (SchNet-style) ties the encoding to the contact scale:

    e_k(d) = exp( -(d - μ_k)^2 / (2 σ^2) ),   k = 1..K

with centers μ_k evenly spaced on [d_min, d_max] and σ = the center spacing.

Calibrated defaults
-------------------
d_min=3.5, d_max=20.0, K=16 come from the empirical edge-distance distribution
(``scripts``-level calibration over the corrected sub-base): p1≈3.7 Å (the
nearest-neighbour Cα peak), median≈7.5 Å, p99≈20 Å. This puts ~8 of 16 centers
in the 4–12 Å contact band (good resolution) while the <1% tail beyond 20 Å
saturates the last RBF (acceptable). The raw scalar distance is kept *alongside*
the expansion by callers (as before).

Centers/width are fixed (non-trainable buffers) by default; ``trainable=True``
makes them learnable (an ablation alternative the plan keeps in mind).
"""
import torch
import torch.nn as nn


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf: int = 16, d_min: float = 3.5, d_max: float = 20.0,
                 trainable: bool = False):
        super().__init__()
        self.num_rbf = num_rbf
        self.trainable = trainable
        centers = torch.linspace(d_min, d_max, num_rbf)
        spacing = float(centers[1] - centers[0]) if num_rbf > 1 else 1.0
        if trainable:
            self.mu = nn.Parameter(centers)
            self.log_sigma = nn.Parameter(torch.tensor(float(spacing)).log())
        else:
            self.register_buffer("mu", centers)
            self.register_buffer("sigma", torch.tensor(float(spacing)))

    def _sigma(self):
        return self.log_sigma.exp() if self.trainable else self.sigma

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """d of any shape ``(...)`` -> ``(..., num_rbf)`` Gaussian features."""
        diff = d.unsqueeze(-1) - self.mu
        return torch.exp(-(diff ** 2) / (2.0 * self._sigma() ** 2 + 1e-12))

    def extra_repr(self) -> str:
        mu = self.mu
        return (f"num_rbf={self.num_rbf}, range=[{float(mu.min()):.2f}, "
                f"{float(mu.max()):.2f}], sigma={float(self._sigma()):.2f}, "
                f"trainable={self.trainable}")

from __future__ import annotations

import torch
from torch import nn


class MLPAEMEncoder(nn.Module):
    def __init__(self, input_dim: int = 70, embed_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

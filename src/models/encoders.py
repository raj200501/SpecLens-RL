"""Small MLP encoders."""

from __future__ import annotations

import torch
from torch import nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(last_dim, hidden), nn.Tanh()])
            last_dim = hidden
        self.net = nn.Sequential(*layers)
        self.output_dim = last_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


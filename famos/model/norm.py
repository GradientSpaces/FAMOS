"""Normalization used by FAMOS attention layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadRMSNorm(nn.Module):
    """RMS-normalize each attention head independently."""

    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        normalized = F.normalize(x.float(), dim=-1, eps=1e-6)
        return (normalized * self.gamma * self.scale).to(original_dtype)

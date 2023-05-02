
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class GeoDiscriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        y = self.head(x)
        return y
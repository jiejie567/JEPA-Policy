import torch
import torch.nn as nn

from mip.networks.chitfm import ChiTransformer


class ChiTransformerDINOAux(ChiTransformer):
    def __init__(self, *args, dino_out_dim: int = 768, **kwargs):
        super().__init__(*args, **kwargs)
        self.future_embed_head = nn.Sequential(
            nn.Linear(self.act_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, dino_out_dim),
        )

    def forward(self, x, s, t, condition=None):
        # Keep original interface for compatibility with existing losses.
        return super().forward(x, s, t, condition)

    def forward_with_aux(self, x, s, t, condition=None):
        y, scalar = super().forward(x, s, t, condition)
        pooled = y.mean(dim=1)  # (B, act_dim)
        future_embed_pred = self.future_embed_head(pooled)
        return y, scalar, future_embed_pred

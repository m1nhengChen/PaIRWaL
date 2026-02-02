from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from record import PAD


class TokenNodeFusion(nn.Module):
    """
    Per-position fusion:
      e = Emb(token_id) + MLP(node_feat) * node_mask
    """
    def __init__(self, vocab_size: int, emb_dim: int, feat_dim: int, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.dropout = dropout

    def forward(self, tok: torch.Tensor, node_feat: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        tok: (B,L) long
        node_feat: (B,L,F) float
        node_mask: (B,L) float {0,1}
        returns: (B,L,E)
        """
        e_tok = self.emb(tok)  # (B,L,E)
        e_node = self.proj(node_feat) * node_mask.unsqueeze(-1)  # (B,L,E)
        x = e_tok + e_node
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

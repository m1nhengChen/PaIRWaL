# models_readers.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_fusion import TokenNodeFusion


def estimate_vocab_size(record_attr: str, roi_dim: int) -> int:
    """
    Node features are continuous -> not in vocab.
    Vocab includes only small structure tokens and optional ROI attribute tokens starting at ATTR_BASE.
    We use a conservative upper bound to accommodate ATTR_BASE + roi_dim.
    """
    # Keep consistent with record.py ATTR_BASE=4096
    STRUCT_VOCAB_BASE = 128
    ATTR_BASE = 4096
    if record_attr == "roi":
        return max(STRUCT_VOCAB_BASE, ATTR_BASE + roi_dim + 8)
    return STRUCT_VOCAB_BASE


class ReaderBoW(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, feat_dim: int, hidden: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.fuse = TokenNodeFusion(vocab_size, emb_dim, feat_dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, tok, node_feat, node_mask, attn_mask):
        x = self.fuse(tok, node_feat, node_mask)  # (B,L,E)
        mask = attn_mask.unsqueeze(-1).to(x.dtype)
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = x.sum(dim=1) / denom
        return self.mlp(pooled)


class ReaderCNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, feat_dim: int, hidden: int, n_classes: int,
                 dropout: float = 0.2, kernels=(3, 5, 7), channels=128):
        super().__init__()
        self.fuse = TokenNodeFusion(vocab_size, emb_dim, feat_dim, dropout)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim, out_channels=channels, kernel_size=k, padding=k // 2)
            for k in kernels
        ])
        self.head = nn.Sequential(
            nn.Linear(len(kernels) * channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.dropout = dropout

    def forward(self, tok, node_feat, node_mask, attn_mask):
        x = self.fuse(tok, node_feat, node_mask)  # (B,L,E)
        x = x.transpose(1, 2)  # (B,E,L)

        feats = []
        for conv in self.convs:
            h = F.relu(conv(x))  # (B,C,L)
            mask = attn_mask.unsqueeze(1)  # (B,1,L)
            h = h.masked_fill(mask == 0, -1e9)
            feats.append(h.max(dim=2).values)  # (B,C)

        z = torch.cat(feats, dim=1)
        return self.head(z)


class ReaderGRU(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, feat_dim: int, hidden: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.fuse = TokenNodeFusion(vocab_size, emb_dim, feat_dim, dropout)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 2 * hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, n_classes),
        )
        self.dropout = dropout

    def forward(self, tok, node_feat, node_mask, attn_mask):
        x = self.fuse(tok, node_feat, node_mask)  # (B,L,E)
        out, _ = self.gru(x)  # (B,L,2H)
        mask = attn_mask.unsqueeze(-1).to(out.dtype)
        out = out * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = out.sum(dim=1) / denom
        return self.head(pooled)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class ReaderTransformer(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, feat_dim: int, hidden: int, n_classes: int,
                 dropout: float = 0.2, n_layers: int = 2, n_heads: int = 4, ff_mult: int = 4):
        super().__init__()
        self.fuse = TokenNodeFusion(vocab_size, emb_dim, feat_dim, dropout)
        self.pe = PositionalEncoding(emb_dim, max_len=8192)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_mult * emb_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.dropout = dropout

    def forward(self, tok, node_feat, node_mask, attn_mask):
        x = self.fuse(tok, node_feat, node_mask)  # (B,L,E)
        x = self.pe(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        key_padding = (attn_mask == 0)  # True for PAD
        h = self.enc(x, src_key_padding_mask=key_padding)  # (B,L,E)
        mask = attn_mask.unsqueeze(-1).to(h.dtype)
        h = h * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = h.sum(dim=1) / denom
        return self.head(pooled)


def build_reader(
    reader: str,
    vocab_size: int,
    emb_dim: int,
    feat_dim: int,
    hidden: int,
    n_classes: int,
    dropout: float,
    tr_layers: int,
    tr_heads: int,
) -> nn.Module:
    r = reader.lower()
    if r == "bow":
        return ReaderBoW(vocab_size, emb_dim, feat_dim, hidden, n_classes, dropout)
    if r == "cnn":
        return ReaderCNN(vocab_size, emb_dim, feat_dim, hidden, n_classes, dropout)
    if r == "gru":
        return ReaderGRU(vocab_size, emb_dim, feat_dim, hidden, n_classes, dropout)
    if r == "transformer":
        return ReaderTransformer(vocab_size, emb_dim, feat_dim, hidden, n_classes, dropout, n_layers=tr_layers, n_heads=tr_heads)
    raise ValueError("Unknown --reader. Choose from: bow, cnn, gru, transformer")

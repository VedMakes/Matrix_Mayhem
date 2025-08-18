from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import MiniGPTConfig

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=2048):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)  # [1, T]
        tok = self.tok_embed(x)       # [B, T, d_model]
        pos = self.pos_embed(pos)     # [1, T, d_model]
        return tok + pos              # [B, T, d_model]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k=None, v=None, mask=None):
        B, L, _ = q.shape
        if k is None: k = q
        if v is None: v = q

        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        Q = Q.view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, L, Dh)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x = x + self.mha(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x

class MiniGPTModel(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        self.embed = EmbeddingLayer(config.vocab_size, config.d_model, max_len=config.max_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.dropout)
             for _ in range(config.num_layers)]
        )
        self.ln_final = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x, mask=None):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_final(x)
        return self.head(x)  # [B, T, V]

    # HF-like API
    def save_pretrained(self, out_dir: str | Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # save config
        self.config.to_json(out_dir / "config.json")
        # save weights
        torch.save(self.state_dict(), out_dir / "pytorch_model.bin")

    @staticmethod
    def from_pretrained(path: str | Path) -> "MiniGPTModel":
        path = Path(path)
        config = MiniGPTConfig.from_json(path / "config.json")
        model = MiniGPTModel(config)
        sd = torch.load(path / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(sd, strict=True)
        return model

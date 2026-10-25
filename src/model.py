"""1-layer transformer for modular addition (Nanda et al. 2023), reimplementation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

P = 113
D_VOCAB = 114  # 0..112 plus "=" token at 113
N_CTX = 3
D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
D_MLP = 512
EQ_TOKEN = 113


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(D_MODEL, D_VOCAB) / math.sqrt(D_MODEL))
        self.W_pos = nn.Parameter(torch.randn(N_CTX, D_MODEL) / math.sqrt(D_MODEL))
        self.W_K = nn.Parameter(torch.randn(N_HEADS, D_HEAD, D_MODEL) / math.sqrt(D_MODEL))
        self.W_Q = nn.Parameter(torch.randn(N_HEADS, D_HEAD, D_MODEL) / math.sqrt(D_MODEL))
        self.W_V = nn.Parameter(torch.randn(N_HEADS, D_HEAD, D_MODEL) / math.sqrt(D_MODEL))
        self.W_O = nn.Parameter(torch.randn(D_MODEL, D_MODEL) / math.sqrt(D_MODEL))
        self.W_in = nn.Parameter(torch.randn(D_MLP, D_MODEL) / math.sqrt(D_MODEL))
        self.b_in = nn.Parameter(torch.zeros(D_MLP))
        self.W_out = nn.Parameter(torch.randn(D_MODEL, D_MLP) / math.sqrt(D_MODEL))
        self.b_out = nn.Parameter(torch.zeros(D_MODEL))
        self.W_U = nn.Parameter(torch.randn(D_MODEL, D_VOCAB) / math.sqrt(D_VOCAB))

    def _attn(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        # q,k,v: (B, H, T, dh)
        q = torch.einsum("h d k, b t k -> b h t d", self.W_Q, x)
        k = torch.einsum("h d k, b t k -> b h t d", self.W_K, x)
        v = torch.einsum("h d k, b t k -> b h t d", self.W_V, x)
        scores = torch.einsum("b h t d, b h s d -> b h t s", q, k) / math.sqrt(D_HEAD)
        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask[None, None, :, :], -1e10)
        pat = torch.softmax(scores, dim=-1)
        z = torch.einsum("b h t s, b h s d -> b h t d", pat, v)
        z = z.permute(0, 2, 1, 3).contiguous().reshape(B, T, D)
        out = z @ self.W_O.T
        return out

    def _mlp(self, x):
        h = x @ self.W_in.T + self.b_in  # (B,T,d_mlp)
        h = F.relu(h)
        out = h @ self.W_out.T + self.b_out
        return out

    def forward(self, tokens):
        # tokens: (B, T) long
        B, T = tokens.shape
        emb = self.W_E[:, tokens].permute(1, 2, 0).contiguous()
        x = emb + self.W_pos[None, :, :]
        x = x + self._attn(x)
        x = x + self._mlp(x)
        logits = x @ self.W_U  # (B, T, V)
        return logits

    def mlp_post_activation(self, tokens):
        B, T = tokens.shape
        emb = self.W_E[:, tokens].permute(1, 2, 0).contiguous()
        x = emb + self.W_pos[None, :, :]
        x = x + self._attn(x)
        h = x @ self.W_in.T + self.b_in
        h = F.relu(h)
        # return last position
        return h[:, -1, :]  # (B, d_mlp)


def make_model(device="cpu"):
    torch.manual_seed(0)
    m = Transformer()
    return m.to(device)

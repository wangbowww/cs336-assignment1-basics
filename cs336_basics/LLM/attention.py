"""
    Attention Implementations
"""
import torch
import torch.nn as nn
from jaxtyping import Float, Bool, Int
from einops import rearrange
import math

from ..utils import softmax
from .rope import RoPE
from .linear_module import Linear

def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scores = Q @ rearrange(K, " ... keys d_k -> ... d_k keys") / math.sqrt(d_k)

    # replace the value to -INF if mask[i,j] == False
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attn = softmax(scores, dim=-1)
    return attn @ V

class MHAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int, 
        theta: float | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.Wqkv = Linear(d_model, 3 * d_model)
        self.Wo = Linear(d_model, d_model)
        self.rope = RoPE(theta=theta, d_model=d_model // num_heads, max_seq_len=max_seq_len) if theta is not None and max_seq_len is not None else None

    def forward(
        self,
        in_features: Float[torch.Tensor, " ... sequence_length d_model"],
        token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
    ) -> Float[torch.Tensor, " ... sequence_length d_out"]:
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, got d_model={self.d_model} and num_heads={self.num_heads}")

        # Compute Q, K, V in one large projection and then split.
        qkv = self.Wqkv(in_features)  # shape: (..., seq_len, 3 * d_model)
        Q, K, V = torch.split(qkv, [self.d_model, self.d_model, self.d_model], dim=-1)
        d_k = self.d_model // self.num_heads
        Q = rearrange(Q, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads, d_k=d_k)
        K = rearrange(K, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads, d_k=d_k)
        V = rearrange(V, "... seq (head d_v) -> ... head seq d_v", head=self.num_heads, d_v=d_k)

        # apply RoPE
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(Q.shape[-2], device=Q.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        # causal mask
        mask = torch.triu(torch.ones(Q.shape[-2], Q.shape[-2], device=Q.device, dtype=torch.bool), diagonal=1)
        mask = ~mask
        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)

        # reshape back and apply output projection
        attn_out = rearrange(attn_out, "... head seq d_v -> ... seq (head d_v)")
        return self.Wo(attn_out)
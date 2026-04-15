"""
    Scaled dot product attention Implementation
"""
import torch
from jaxtyping import Float, Bool, Int
from einops import rearrange
import math

from .softmax import softmax
from .rope import RoPE

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

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[torch.Tensor, " d_k d_model"],
    k_proj_weight: Float[torch.Tensor, " d_k d_model"],
    v_proj_weight: Float[torch.Tensor, " d_v d_model"],
    o_proj_weight: Float[torch.Tensor, " d_model d_v"],
    in_features: Float[torch.Tensor, " ... sequence_length d_model"],
    theta: float | None = None,
    max_seq_len: int | None = None,
    token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
) -> Float[torch.Tensor, " ... sequence_length d_out"]:
    if d_model % num_heads != 0:
        raise ValueError(f"d_model must be divisible by num_heads, got d_model={d_model} and num_heads={num_heads}")

    q_total = q_proj_weight.shape[0]
    k_total = k_proj_weight.shape[0]
    v_total = v_proj_weight.shape[0]
    if q_total % num_heads != 0 or k_total % num_heads != 0 or v_total % num_heads != 0:
        raise ValueError("Q/K/V projection output dims must be divisible by num_heads")

    d_k = q_total // num_heads
    if k_total // num_heads != d_k:
        raise ValueError("Per-head Q and K dimensions must match")
    d_v = v_total // num_heads

    # Compute Q, K, V in one large projection and then split.
    qkv_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    qkv = in_features @ qkv_proj_weight.T
    Q, K, V = torch.split(qkv, [q_total, k_total, v_total], dim=-1)

    Q = rearrange(Q, "... seq (head d_k) -> ... head seq d_k", head=num_heads, d_k=d_k)
    K = rearrange(K, "... seq (head d_k) -> ... head seq d_k", head=num_heads, d_k=d_k)
    V = rearrange(V, "... seq (head d_v) -> ... head seq d_v", head=num_heads, d_v=d_v)

    # apply RoPE
    if theta is not None and max_seq_len is not None and token_positions is not None:
        rope = RoPE(theta=theta, d_model=d_k, max_seq_len=max_seq_len, device=in_features.device)
        Q = rope(Q, token_positions)
        K = rope(K, token_positions)
    # causal mask
    mask = torch.triu(torch.ones(Q.shape[-2], Q.shape[-2], device=Q.device, dtype=torch.bool), diagonal=1)
    mask = ~mask
    attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)

    # reshape back and apply output projection
    attn_out = rearrange(attn_out, "... head seq d_v -> ... seq (head d_v)")
    return attn_out @ o_proj_weight.T

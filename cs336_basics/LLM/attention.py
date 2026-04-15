"""
    Scaled dot product attention Implementation
"""
import torch
from jaxtyping import Float, Bool
from einops import rearrange
import math

from .softmax import softmax

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

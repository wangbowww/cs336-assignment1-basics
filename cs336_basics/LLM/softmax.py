"""
    Softmax Implementation
"""
import torch
from jaxtyping import Float

def softmax(
    in_features: Float[torch.Tensor, " ..."],
    dim: int
) -> Float[torch.Tensor, " ..."]:
    mx = torch.max(in_features, dim=dim, keepdim=True).values
    shifted = in_features - mx
    exp_shifted = torch.exp(shifted)
    sm = torch.sum(exp_shifted, dim=dim, keepdim=True)
    return exp_shifted / sm
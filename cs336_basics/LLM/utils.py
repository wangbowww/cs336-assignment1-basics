"""
Utility Functions for LLM Modules
"""
import torch

def init_linear_weight(
    W: torch.Tensor,
    in_features: int,
    out_features: int
) -> torch.Tensor:
    std = (2 / (in_features + out_features)) ** 0.5
    torch.nn.init.trunc_normal_(W, mean=0.0, std=std, a=-3*std, b=3*std)
    return W

def init_embedding_weight(
    W: torch.Tensor
) -> torch.Tensor:
    torch.nn.init.trunc_normal_(W, mean=0.0, std=1, a=-3, b=3)
    return W

def init_RMSNorm_weight(
    W: torch.Tensor
) -> torch.Tensor:
    raise NotImplementedError
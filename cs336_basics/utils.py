"""
Utility Functions for cs336 basics
"""
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch import arange
from torch import logsumexp

"""
    weight initialization
"""
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
    return torch.nn.init.ones_(W)


"""
    softmax function
"""
def softmax(
    in_features: Float[torch.Tensor, " ..."],
    dim: int
) -> Float[torch.Tensor, " ..."]:
    mx = torch.max(in_features, dim=dim, keepdim=True).values
    shifted = in_features - mx
    exp_shifted = torch.exp(shifted)
    sm = torch.sum(exp_shifted, dim=dim, keepdim=True)
    return exp_shifted / sm

"""
    loss function
"""
def cross_entropy_loss(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]    
) -> Float[Tensor, ""]:
    """
    Compute the cross-entropy loss between predictions and targets.

    Returns:
        The average cross-entropy loss across the batch.
    """
    probs = inputs - logsumexp(inputs, dim=-1, keepdim=True)  # Convert logits to probabilities
    batch_size = inputs.shape[-2]
    loss = -probs[arange(batch_size), targets].mean()  # Average negative log-likelihood
    return loss

"""
    Silu function
"""
def silu(
    x: Float[torch.Tensor, " ... d_model"]
) -> Float[torch.Tensor, " ... d_model"]:
    return x * torch.sigmoid(x)
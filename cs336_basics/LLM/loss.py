"""
    Loss functions for LLM training.
"""
from jaxtyping import Float, Int
from torch import Tensor
from torch import arange
from torch import logsumexp

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
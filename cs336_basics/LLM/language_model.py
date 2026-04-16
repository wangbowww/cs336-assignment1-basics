"""
    Transforer LM Implementation
"""
import torch
import torch.nn as nn
from jaxtyping import Float, Int

from .embedding_module import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .linear_module import Linear


class LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.input_embedding = Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.out_embedding = Linear(d_model, vocab_size)

    def forward(
        self, 
        in_indices: Int[torch.Tensor, " batch_size seq"],
    ) -> Float[torch.Tensor, " batch_size seq vocab_size"]:
        # unnormalized output, i.e. logits
        return self.out_embedding(
            self.norm(
                self.blocks(
                    self.input_embedding(in_indices)
                )
            )
        )
"""
    Transformer Block Implementation
"""
import torch
import torch.nn as nn
from jaxtyping import Float

from .rope import RoPE
from .rmsnorm import RMSNorm
from .attention import MHAttention
from .ffn import SwiGLUFFN

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)
        self.attention = MHAttention(d_model, num_heads, theta, max_seq_len)

    def forward(
        self,
        in_features: Float[torch.Tensor, " batch sequence_length d_model"],
    ) -> Float[torch.Tensor, " batch sequence_length d_model"]:
        out_attn = self.attention(self.norm1(in_features)) + in_features
        return self.ffn(self.norm2(out_attn)) + out_attn
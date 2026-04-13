"""
    Position-wise FFN implementation
"""
import torch
import torch.nn as nn
from .linear_module import Linear
from jaxtyping import Float

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.W1 = Linear(d_model, d_ff, device, dtype)
        self.W2 = Linear(d_ff, d_model, device, dtype)
        self.W3 = Linear(d_model, d_ff, device, dtype)
    
    @staticmethod
    def silu(
        x: Float[torch.Tensor, " ... d_model"]
    ) -> Float[torch.Tensor, " ... d_model"]:
        return x * torch.sigmoid(x)

    def forward(
        self,
        x: Float[torch.Tensor, " ... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        return self.W2(self.silu(self.W1(x)) * self.W3(x))
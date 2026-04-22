"""
    RMSNorm implementation
"""
import torch
import torch.nn as nn
from jaxtyping import Float
from einops import reduce
from ..utils import init_RMSNorm_weight

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        w = torch.empty((d_model,), device=self.device, dtype=self.dtype)
        init_RMSNorm_weight(w)
        self.gain = nn.Parameter(w)

    def forward(
        self,
        x: Float[torch.Tensor, " ... d_model"]
    ) -> Float[torch.Tensor, " ... d_model"]:
        in_dtype = x.dtype
        # prevent overflow
        x.to(torch.float32)
        # input x*x, and reduce the last dim using "mean"
        mean_square = reduce(x * x, "... d_model -> ... 1", "mean")
        # x / ...(broadcast to (...,d))
        result = x / torch.sqrt(mean_square + self.eps) * self.gain
        return result.to(in_dtype)
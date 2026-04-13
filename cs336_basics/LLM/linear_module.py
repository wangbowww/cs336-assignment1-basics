"""
    Linear Module implementation
"""
import torch
from torch import nn
from jaxtyping import Float
from .utils import init_linear_weight

class Linear(nn.Module):
    def __init__(
        self,
        d_in: int, # final dimension of the input
        d_out: int, # final dimension of the output
        device: torch.device | None = None, # Device to store the parameters on
        dtype: torch.dtype | None = None # Data type of the parameters
    ):
        # Construct a linear transformation module.
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype
        w = torch.empty(
            self.d_out,
            self.d_in,
            device=self.device,
            dtype=self.dtype)
        init_linear_weight(w, self.d_in, self.d_out)
        self.W = nn.Parameter(w)

    def forward(self,
        x: Float[torch.Tensor, " ... d_in"]
    ) -> Float[torch.Tensor, " ... d_out"]:
        # Apply the linear transformation to the input
        return x @ self.W.T
"""
    Linear Module implementation
"""
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int, # final dimension of the input
        out_features: int, # final dimension of the output
        device: torch.device | None = None, # Device to store the parameters on
        dtype: torch.dtype | None = None # Data type of the parameters
    ):
        # Construct a linear transformation module.
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        w = torch.empty(
            self.out_features,
            self.in_features,
            device=self.device,
            dtype=self.dtype)
        std = 2/(self.in_features + self.out_features) ** 0.5
        nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3*std, b=3*std)
        self.W = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the linear transformation to the input
        return x @ self.W.T
"""
    RoPE implementation
"""
import torch
import torch.nn as nn
from jaxtyping import Float, Int

class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_model: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        if d_model % 2 != 0:
            raise ValueError(f"RoPE's d_model must be even, got: {d_model}")
        super().__init__()
        self.theta = theta
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device
        # shape of angles: [max_seq_len, d_model // 2]
        angles = [
            [i / self.theta ** ((2 * k - 2) / self.d_model)
                for k in range(1, self.d_model // 2 + 1)
            ]
                for i in range(0, self.max_seq_len)
            ]
        cos_cached: Float[torch.Tensor, " max_seq_len, half_d_model"] = torch.tensor(angles, device=self.device, dtype=torch.float32)
        sin_cached: Float[torch.Tensor, " max_seq_len, half_d_model"] = torch.tensor(angles, device=self.device, dtype=torch.float32)
        cos_cached = torch.cos(cos_cached)
        sin_cached = torch.sin(sin_cached)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, " ... seq_len d_model"],
        token_positions: Int[torch.Tensor, " ... seq_len"]
    ) -> Float[torch.Tensor, " ... seq_len d_model"]:
        cos = self.cos_cached[token_positions]  # pyright: ignore[reportIndexIssue]
        sin = self.sin_cached[token_positions]  # pyright: ignore[reportIndexIssue]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        result = x.clone()
        result[..., 0::2] = x_even * cos - x_odd * sin
        result[..., 1::2] = x_even * sin + x_odd * cos
        return result
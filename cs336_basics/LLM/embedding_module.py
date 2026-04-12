"""
    Embedding Module implementation
"""
import torch
import torch.nn as nn
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int, # size of the vocabulary
        embedding_dim: int, # dimension of hidden size
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.device = device
        self.dtype = dtype
        w = torch.empty(size=(self.vocab_size, self.d_model), device=device, dtype=dtype)
        std = 2/(self.vocab_size + self.d_model) ** 0.5
        nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3*std, b=3*std)
        self.embeddings = nn.Parameter(w)
    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.embeddings[token_ids]
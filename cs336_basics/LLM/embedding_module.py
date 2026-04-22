"""
    Embedding Module implementation
"""
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from ..utils import init_embedding_weight

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
        init_embedding_weight(w)
        self.embeddings = nn.Parameter(w)
    def forward(
        self,
        token_ids: Int[torch.Tensor, " ..."]
    ) -> Float[torch.Tensor, " ... d_model"]:
        return self.embeddings[token_ids]
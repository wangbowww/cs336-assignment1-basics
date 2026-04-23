"""
    model config (hyperparameters), default is GPT2-XL config
"""
from dataclasses import dataclass

@dataclass
class model_config:
    # input seq len
    seq_len: int = 1024

    # vocabulary size, for token embedding
    vocab_size: int = 50257 

    # hidden dimension of token embedding
    d_model: int = 1600 

    # d_ff in FFN
    d_ff: int = 6400

    # num of attn heads
    num_heads: int = 25

    # num of Transformer layers
    num_layers: int = 48 

    # per data storage
    data_type_size: int = 4
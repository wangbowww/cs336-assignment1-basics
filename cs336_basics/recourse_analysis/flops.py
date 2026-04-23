"""
    return flops for every components in LLM training/inference
"""
from collections import OrderedDict
from .model_config import model_config

def matmul_flops(n, m, p):
    """
        get the flops for a matrix multiplication of size (n, m) and (m, p)
    """
    return n * m * p * 2

def input_embedding():
    """
        get the flops for input embedding
    """
    # cause it index the embedding matrix, rather than matmul
    return 0

def norm():
    """
        get the flops for a layer norm of size d_model
    """
    # cause it only do element-wise operation, rather than matmul
    return 0

def rope():
    '''
        get the flops for RoPE
    '''
    # cause it only do element-wise operation, rather than matmul
    return 0

def ffn_per_layer(seq_len, d_model, d_ff):
    """
        get the flops for a feed forward network
    """
    # silu is a simple element-wise operation, so we can ignore it
    # W1 * W3 is also a simple element-wise operation, so we can ignore it
    # (seq_len, d_model) x (d_model, d_ff) -> (seq_len, d_ff)
    ffw1_flops = matmul_flops(seq_len, d_model, d_ff)
    ffw3_flops = matmul_flops(seq_len, d_model, d_ff)
    # (seq_len, d_ff) x (d_ff, d_model) -> (seq_len, d_model)
    ffw2_flops = matmul_flops(seq_len, d_ff, d_model)
    return ffw1_flops + ffw2_flops + ffw3_flops

def attention_per_layer(seq_len, d_model, num_heads):
    """
        get the flops for a multi-head attention
    """
    head_dim = d_model // num_heads
    # Q, K, V projections
    # (seq_len, d_model) x (d_model, 3 * d_model) -> (seq_len, 3 * d_model)
    qkv_flops = matmul_flops(seq_len, d_model, 3 * d_model)
    # RoPE is a simple element-wise operation, so we can ignore it

    # attention scores
    # (seq_len, head_dim) x (head_dim, seq_len) -> (seq_len, seq_len) qk, and "/sqrt(d_k)" is ignored
    # (seq_len, seq_len) x (seq_len, head_dim) -> (seq_len, head_dim) scoresv
    # softmax is also ignored since it's a simple element-wise operation
    attn_scores_flops = (
        matmul_flops(seq_len, head_dim, seq_len)
        + matmul_flops(seq_len, seq_len, head_dim)
    ) * num_heads

    # attention output
    # (seq_len, d_model) x (d_model, d_model) -> (seq_len, d_model)
    attn_output_flops = matmul_flops(seq_len, d_model, d_model)
    return qkv_flops + attn_scores_flops + attn_output_flops
    
def out_embedding(seq_len, d_model, vocab_size):
    """
        get the flops for output embedding
    """
    # (seq_len, d_model) x (d_model, vocab_size) -> (seq_len, vocab_size)
    return matmul_flops(seq_len, d_model, vocab_size)

def model_forward_flops_counting(
    model_config: model_config
):
    """
        get the total flops for a forward pass of a transformer model
    """
    seq_len = model_config.seq_len
    d_model = model_config.d_model
    num_heads = model_config.num_heads
    d_ff = model_config.d_ff
    num_layers = model_config.num_layers
    vocab_size = model_config.vocab_size

    result = OrderedDict()

    # rope
    result["rope(no_trainable)"] = rope() * num_layers

    # input embedding
    result["input_embedding"] = input_embedding()

    # norm in transformer block, there are 2 norms in each transformer block
    result["norm_in_transformer_block"] = norm() * num_layers * 2

    # attention
    result["attention"] = attention_per_layer(seq_len, d_model, num_heads) * num_layers

    # feed forward
    result["ffn"] = ffn_per_layer(seq_len, d_model, d_ff) * num_layers

    # final norm
    result["final_norm"] = norm() * num_layers

    # output embedding
    result["output_embedding"] = out_embedding(seq_len, d_model, vocab_size)

    result["total_params"] = sum(result.values())

    return result
"""
    return memory footprint for every components in LLM training/inference
"""
from .model_config import model_config
from collections import OrderedDict

def bytes_to_GB(bytes: int):
    return bytes / (1024 ** 3)

def matmul_mem(n, m, data_type_size):
    """
        get the memory footprint for a matrix size (n, m)
    """
    return bytes_to_GB(n * m * data_type_size)

def input_embedding(vocab_size, d_model, data_type_size):
    """
        get memory footprint for input embedding
    """
    return matmul_mem(vocab_size, d_model, data_type_size)

def norm(d_model, data_type_size):
    '''
        get memory footprint for a layer norm of size d_model
    '''
    return matmul_mem(d_model, 1, data_type_size)

def ffn_per_layer(d_model, d_ff, data_type_size):
    '''
        get memory footprint for a feed forward network with input size d_model and hidden size d_ff
    '''
    # W1, W2, W3 are the parameters of the swiglu ffn
    return matmul_mem(d_model, d_ff, data_type_size) + matmul_mem(d_ff, d_model, data_type_size) + matmul_mem(d_model, d_ff, data_type_size)

def attention_per_layer(d_model, data_type_size):
    '''
        get memory footprint for a multi-head attention with input size d_model
    '''
    # we assume d_model // num_heads == head_dim
    # Q, K, V, O projections (d_model, d_model) + RoPE(but rope is in buffer, not trainable parameter, so we ignore it here)
    return matmul_mem(d_model, d_model, data_type_size) * 4
def out_embedding(vocab_size, d_model, data_type_size):
    '''
        get memory footprint for output embedding
    '''
    return matmul_mem(d_model, vocab_size, data_type_size)

def rope(vocab_size, d_model, data_type_size):
    '''
        get memory footprint for RoPE
    '''
    return matmul_mem(vocab_size, d_model, data_type_size)

def model_weights_counting(
    model_config: model_config
) -> OrderedDict:
    """
        get the memory footprint of parameters for a transformer model
    """
    d_model = model_config.d_model
    d_ff = model_config.d_ff
    num_layers = model_config.num_layers
    vocab_size = model_config.vocab_size
    dtype_size = model_config.data_type_size

    result = OrderedDict()

    # rope is not trainable parameter, but we still want to count it for memory footprint
    result["rope(no_trainable)"] = rope(vocab_size, d_model, dtype_size)

    # input embedding
    result["input_embedding"] = input_embedding(vocab_size, d_model, dtype_size)

    # norm in transformer block, there are 2 norms in each transformer block
    result["norm_in_transformer_block"] = norm(d_model, dtype_size) * num_layers * 2
    # attention
    result["attention"] = attention_per_layer(d_model, dtype_size) * num_layers

    # feed forward
    result["ffn"] = ffn_per_layer(d_model, d_ff, dtype_size) * num_layers

    # final norm
    result["final_norm"] = norm(d_model, dtype_size)

    # output embedding
    result["output_embedding"] = out_embedding(vocab_size, d_model, dtype_size)

    result["total_params"] = sum(result.values())

    return result

def model_parameters_counting(
    model_config: model_config
) -> OrderedDict:
    result = model_weights_counting(model_config)
    for key in result:
        result[key] = result[key] * (1024 ** 3) / model_config.data_type_size
    result["trainable_params"] = result["total_params"] - result["rope(no_trainable)"]
    return result
"""
    resource counting for LLMs
"""
from dataclasses import dataclass
from collections import OrderedDict

import matplotlib.pyplot as plt

@dataclass
class model_config:
    seq_len: int = 1024
    d_model: int = 1600
    num_heads: int = 25
    d_ff: int = 6400
    num_layers: int = 48
    vocab_size: int = 50257
    data_type_size: int = 4  # float32

def matrixFlops(n, m, p):
    """
        get the flops for a matrix multiplication of size (n, m) and (m, p)
    """
    return n * m * p * 2


def attentionFlopsPerLayer(seq_len, d_model, num_heads):
    """
        get the flops for attention
    """
    head_dim = d_model // num_heads
    # Q, K, V projections
    # (seq_len, d_model) x (d_model, 3 * d_model) -> (seq_len, 3 * d_model)
    qkv_flops = matrixFlops(seq_len, d_model, 3 * d_model)
    # RoPE is a simple element-wise operation, so we can ignore it

    # attention scores
    # (seq_len, head_dim) x (head_dim, seq_len) -> (seq_len, seq_len) qk, and "/sqrt(d_k)" is ignored
    # (seq_len, seq_len) x (seq_len, head_dim) -> (seq_len, head_dim) scoresv
    # softmax is also ignored since it's a simple element-wise operation
    attn_scores_flops = (
        matrixFlops(seq_len, head_dim, seq_len)
        + matrixFlops(seq_len, seq_len, head_dim)
    ) * num_heads

    # attention output
    # (seq_len, d_model) x (d_model, d_model) -> (seq_len, d_model)
    attn_output_flops = matrixFlops(seq_len, d_model, d_model)
    return qkv_flops + attn_scores_flops + attn_output_flops


def feedForwardFlopsPerLayer(seq_len, d_model, d_ff):
    """
        get the flops for feed forward network
    """
    # silu is a simple element-wise operation, so we can ignore it
    # W1 * W3 is also a simple element-wise operation, so we can ignore it
    # (seq_len, d_model) x (d_model, d_ff) -> (seq_len, d_ff)
    ffw1_flops = matrixFlops(seq_len, d_model, d_ff)
    ffw3_flops = matrixFlops(seq_len, d_model, d_ff)
    # (seq_len, d_ff) x (d_ff, d_model) -> (seq_len, d_model)
    ffw2_flops = matrixFlops(seq_len, d_ff, d_model)
    return ffw1_flops + ffw2_flops + ffw3_flops


def model_weights_counting(
    model_config: model_config
):
    """
        get the memory footprint of parameters for a transformer model
    """
    d_model = model_config.d_model
    d_ff = model_config.d_ff
    num_layers = model_config.num_layers
    vocab_size = model_config.vocab_size
    dtype_size = model_config.data_type_size

    result = OrderedDict()
    # input embedding
    input_embedding_params = d_model * vocab_size
    result["input_embedding"] = input_embedding_params
    result["input_embedding_gb"] = bytes_to_gb(input_embedding_params * dtype_size)

    # attention
    qkvo_params = d_model * d_model * 4  # Q, K, V, O projections
    norm_params = d_model * 2  # 2 RMSNorm layers per block, each with weight
    attn_output_params = qkvo_params + norm_params
    attn_output_params = attn_output_params * num_layers
    result["attention"] = attn_output_params
    result["attention_gb"] = bytes_to_gb(attn_output_params * dtype_size)

    # feed forward
    W1_params = d_model * d_ff
    W2_params = d_ff * d_model
    W3_params = d_model * d_ff
    ffn_params = W1_params + W2_params + W3_params
    ffn_params = ffn_params * num_layers
    result["ffn"] = ffn_params
    result["ffn_gb"] = bytes_to_gb(ffn_params * dtype_size)

    # final norm
    final_norm_params = d_model
    result["final_norm"] = final_norm_params
    result["final_norm_gb"] = bytes_to_gb(final_norm_params * dtype_size)

    # output embedding
    output_embedding_params = d_model * vocab_size
    result["output_embedding"] = output_embedding_params
    result["output_embedding_gb"] = bytes_to_gb(output_embedding_params * dtype_size)

    total_params = (
        input_embedding_params
        + attn_output_params
        + ffn_params
        + final_norm_params
        + output_embedding_params
    )
    result["total_params"] = total_params
    result["total_gb"] = bytes_to_gb(total_params * dtype_size)

    return result

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
    # input embedding
    input_embedding_flops = 0  # embedding lookup is not a matrix multiplication, so we can ignore it
    result["input_embedding"] = input_embedding_flops

    # attention
    attention_flops = attentionFlopsPerLayer(seq_len, d_model, num_heads) * num_layers
    result["attention"] = attention_flops

    # feed forward
    feed_forward_flops = feedForwardFlopsPerLayer(seq_len, d_model, d_ff) * num_layers
    result["ffn"] = feed_forward_flops

    # output embedding
    output_embedding_flops = matrixFlops(seq_len, d_model, vocab_size)
    result["output_embedding"] = output_embedding_flops

    total_flops = (
        input_embedding_flops + attention_flops + feed_forward_flops + output_embedding_flops
    )
    result["total_flops"] = total_flops

    return result

def bytes_to_gb(bytes):
    return bytes / (1024 ** 3)

# generate by codex (GPT 5.4)
def print_ordered_result(result: OrderedDict, keys=None, value_format=".2e"):
    """
        Print result components in a fixed order.
    """
    if keys is None:
        keys = list(result.keys())

    for key in keys:
        if key not in result:
            continue
        value = result[key]
        if isinstance(value, float):
            print(f"{key}: {value:{value_format}}")
        else:
            print(f"{key}: {value}")

# generate by codex (GPT 5.4)
def analyze_gpt2_xl_long_context_flops(
    config: model_config,
):
    """
        Compare GPT-2 XL FLOPs at the default context length and at 16,384 tokens.
    """
    base_config = model_config(
        seq_len=config.seq_len,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        data_type_size=config.data_type_size,
    )
    long_context_config = model_config(
        seq_len=16384,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        data_type_size=config.data_type_size,
    )

    base_result = model_forward_flops_counting(base_config)
    long_context_result = model_forward_flops_counting(long_context_config)

    component_names = ["input_embedding", "attention", "ffn", "output_embedding"]
    base_proportions = OrderedDict()
    long_context_proportions = OrderedDict()

    for component_name in component_names:
        base_proportions[component_name] = base_result[component_name] / base_result["total_flops"]
        long_context_proportions[component_name] = long_context_result[component_name] / long_context_result["total_flops"]

    total_flops_ratio = long_context_result["total_flops"] / base_result["total_flops"]

    print(f"GPT-2 XL context scaling analysis ({base_config.seq_len} -> {long_context_config.seq_len})")
    print("Base context FLOPs:")
    print_ordered_result(base_result, [*component_names, "total_flops"])
    print()
    print("Long context FLOPs:")
    print_ordered_result(long_context_result, [*component_names, "total_flops"])
    print()
    print(f"Total FLOPs increase: {total_flops_ratio:.2f}x")
    print("Component proportions at base context:")
    for component_name in component_names:
        print(f"{component_name}: {base_proportions[component_name]:.2%}")
    print()
    print("Component proportions at 16,384 context:")
    for component_name in component_names:
        print(f"{component_name}: {long_context_proportions[component_name]:.2%}")
    print()
    print(
        "Summary: Increasing GPT-2 XL from 1,024 to 16,384 tokens increases total forward FLOPs by "
        f"{total_flops_ratio:.2f}x. The attention share grows from {base_proportions['attention']:.2%} "
        f"to {long_context_proportions['attention']:.2%}, while the FFN share drops from "
        f"{base_proportions['ffn']:.2%} to {long_context_proportions['ffn']:.2%}."
    )
    print(
        "The output embedding also becomes proportionally smaller "
        f"({base_proportions['output_embedding']:.2%} -> {long_context_proportions['output_embedding']:.2%}) "
        "because it scales linearly with sequence length, while attention contains a quadratic sequence-length term."
    )

    return {
        "base_result": base_result,
        "long_context_result": long_context_result,
        "base_proportions": base_proportions,
        "long_context_proportions": long_context_proportions,
        "total_flops_ratio": total_flops_ratio,
    }

# generate by codex (GPT 5.4)
def analyze_gpt2_model_scaling_flops(
    config: model_config,
    save_path="gpt2_flops_breakdown.png",
):
    """
        Analyze FLOPs proportions for GPT-2 small / medium / large and save a bar chart.
    """
    small_config = model_config(
        seq_len=config.seq_len,
        d_model=768,
        num_heads=12,
        d_ff=768 * 4,
        num_layers=12,
        vocab_size=config.vocab_size,
        data_type_size=config.data_type_size,
    )
    medium_config = model_config(
        seq_len=config.seq_len,
        d_model=1024,
        num_heads=16,
        d_ff=1024 * 4,
        num_layers=24,
        vocab_size=config.vocab_size,
        data_type_size=config.data_type_size,
    )
    large_config = model_config(
        seq_len=config.seq_len,
        d_model=1280,
        num_heads=20,
        d_ff=1280 * 4,
        num_layers=36,
        vocab_size=config.vocab_size,
        data_type_size=config.data_type_size,
    )

    model_results = OrderedDict(
        [
            ("GPT-2 small", model_forward_flops_counting(small_config)),
            ("GPT-2 medium", model_forward_flops_counting(medium_config)),
            ("GPT-2 large", model_forward_flops_counting(large_config)),
        ]
    )

    component_names = ["input_embedding", "attention", "ffn", "output_embedding"]
    proportion_results = OrderedDict()

    print(f"GPT-2 FLOPs breakdown for seq_len={config.seq_len}, vocab_size={config.vocab_size}")
    print()
    for model_name, result in model_results.items():
        total_flops = result["total_flops"]
        proportion_results[model_name] = OrderedDict()
        print(model_name)
        print(f"  Total FLOPs: {total_flops:.2e}")
        for component_name in component_names:
            flops = result[component_name]
            proportion = flops / total_flops if total_flops else 0
            proportion_results[model_name][component_name] = proportion
            print(f"  {component_name}: {flops:.2e} ({proportion:.2%})")
        print()

    small_prop = proportion_results["GPT-2 small"]
    large_prop = proportion_results["GPT-2 large"]
    print("Summary:")
    print(
        "  As model size increases, the FFN takes a larger share of total forward FLOPs "
        f"({small_prop['ffn']:.2%} -> {large_prop['ffn']:.2%}), and attention also grows slightly "
        f"({small_prop['attention']:.2%} -> {large_prop['attention']:.2%})."
    )
    print(
        "  The output embedding takes proportionally less FLOPs as the model gets larger "
        f"({small_prop['output_embedding']:.2%} -> {large_prop['output_embedding']:.2%}), while input embedding stays negligible because lookup cost is treated as 0 FLOPs here."
    )

    model_names = list(model_results.keys())
    x_positions = list(range(len(model_names)))
    bar_width = 0.18
    offsets = {
        "input_embedding": -1.5 * bar_width,
        "attention": -0.5 * bar_width,
        "ffn": 0.5 * bar_width,
        "output_embedding": 1.5 * bar_width,
    }

    plt.figure(figsize=(10, 6))
    for component_name in component_names:
        component_values = [
            proportion_results[model_name][component_name] * 100
            for model_name in model_names
        ]
        plt.bar(
            [x + offsets[component_name] for x in x_positions],
            component_values,
            width=bar_width,
            label=component_name,
        )

    plt.xticks(x_positions, model_names)
    plt.ylabel("FLOPs (% of total forward pass)")
    plt.title(f"GPT-2 Forward FLOPs Breakdown (seq_len={config.seq_len})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return {
        "model_results": model_results,
        "proportion_results": proportion_results,
    }


if __name__ == "__main__":
    config = model_config()

    print(f"a) Transformer Memory Footprint({config.data_type_size} bytes per parameter):")
    result = model_weights_counting(config)
    print(f"Total parameters: {result['total_params']} ({result['total_gb']:.2f} GB)")
    print("------------------------------------------------------------------------------")

    print("b) Transformer FLOPs:")
    result = model_forward_flops_counting(config)
    print(f"Total FLOPs: {result['total_flops']:.2e}")

    print("------------------------------------------------------------------------------")
    print("c) FFN takes the most FLOPs in transformer")

    print("------------------------------------------------------------------------------")
    print("d) GPT-2 scaling FLOPs breakdown, see the picture saved in gpt2_flops_breakdown.png")
    analyze_gpt2_model_scaling_flops(config)

    print("------------------------------------------------------------------------------")
    print("e) GPT-2 XL long-context FLOPs analysis")
    analyze_gpt2_xl_long_context_flops(config)

    print("------------------------------------------------------------------------------")
    print("Summary:"
    "when the context length is relatively small (e.g., 1024 tokens), "
    "the FFN is the dominant FLOPs contributor, taking around 60-70% of total forward pass FLOPs, "
    "while attention takes around 20-30%."
    "However, as we increase the context length to 16,384 tokens, "
    "the attention FLOPs grow quadratically with sequence length, "
    "becoming the dominant contributor at around 80-90% of total FLOPs, "
    "while the FFN share drops to around 10-20%.")
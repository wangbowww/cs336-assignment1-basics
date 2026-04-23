"""
    resource counting for gpt2-XL, memory footprint and flops
"""
from collections import OrderedDict
import matplotlib.pyplot as plt

from .flops import model_forward_flops_counting

from .memory import model_weights_counting, model_parameters_counting

from .model_config import model_config


class gpt2_xl_config(model_config):
    seq_len: int = 1024
    d_model: int = 1600
    num_heads: int = 25
    d_ff: int = 6400
    num_layers: int = 48
    vocab_size: int = 50257
    data_type_size: int = 4  # float32


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
    config = gpt2_xl_config(data_type_size=2) # single-precision float16 for smaller memory footprint

    result = model_parameters_counting(config)
    print(f"a) Trainable parameters: {result['trainable_params']}")
    result = model_weights_counting(config)
    print(f"Memory footprint of model(single-precision): {result['total_params']:.2e} GB")
    print("------------------------------------------------------------------------------")

    result = model_forward_flops_counting(config)
    print(f"b) Transformer forward FLOPs: {result['total_params']:.2e}")
    print("------------------------------------------------------------------------------")

    print("c) FFN takes the most FLOPs in transformer")
    print("------------------------------------------------------------------------------")
    
    print("d) GPT-2 scaling FLOPs breakdown, see the picture saved in gpt2_flops_breakdown.png")
    # analyze_gpt2_model_scaling_flops(config)
    print("------------------------------------------------------------------------------")
    
    print("e) GPT-2 XL long-context FLOPs analysis")
    # analyze_gpt2_xl_long_context_flops(config)
    print("------------------------------------------------------------------------------")
    
    # print("Summary:"
    # "when the context length is relatively small (e.g., 1024 tokens), "
    # "the FFN is the dominant FLOPs contributor, taking around 60-70% of total forward pass FLOPs, "
    # "while attention takes around 20-30%."
    # "However, as we increase the context length to 16,384 tokens, "
    # "the attention FLOPs grow quadratically with sequence length, "
    # "becoming the dominant contributor at around 80-90% of total FLOPs, "
    # "while the FFN share drops to around 10-20%.")
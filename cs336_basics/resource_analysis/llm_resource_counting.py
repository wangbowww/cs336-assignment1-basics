"""
    resource counting for gpt2-XL, memory footprint and flops
"""
from collections import OrderedDict

from .flops import model_forward_flops_counting

from .memory import model_weights_counting, model_parameters_counting

from .model_config import model_config

from .utils import print_ordered_result, draw_model_flops_breakdown

# generate by codex (GPT 5.4)
def analyze_gpt2_xl_long_context_flops(
    config: model_config,
):
    """
        Compare GPT-2 XL FLOPs at the default context length and at 16,384 tokens.
    """
    base_config = model_config(seq_len=1024)
    long_context_config = model_config(seq_len=16384)

    base_result = model_forward_flops_counting(base_config)
    long_context_result = model_forward_flops_counting(long_context_config)

    component_names = ["attention", "ffn", "output_embedding"]
    base_proportions = OrderedDict()
    long_context_proportions = OrderedDict()

    for component_name in component_names:
        base_proportions[component_name] = base_result[component_name] / base_result["total_flops"]
        long_context_proportions[component_name] = long_context_result[component_name] / long_context_result["total_flops"]

    total_flops_ratio = long_context_result["total_flops"] / base_result["total_flops"]

    print(f"GPT-2 XL context scaling analysis ({base_config.seq_len} -> {long_context_config.seq_len})")
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

if __name__ == "__main__":
    config_gpt2_xl = model_config(data_type_size=2)

    result = model_parameters_counting(config_gpt2_xl)
    print(f"a) Trainable parameters: {result['trainable_params']}")
    result = model_weights_counting(config_gpt2_xl)
    print(f"Memory footprint of model(single-precision): {result['total_params']:.2e} GB")
    print("------------------------------------------------------------------------------")

    result = model_forward_flops_counting(config_gpt2_xl)
    print(f"b) Transformer forward FLOPs: {result['total_flops']:2e}")
    print("------------------------------------------------------------------------------")

    ffn_percent = result['ffn'] / result['total_flops'] * 100
    print(f"c) FFN takes the most FLOPs in gpt2-XL: {ffn_percent:.2f}%")
    print("------------------------------------------------------------------------------")
    
    config_gpt2_small = model_config(num_layers=12, d_model=768, d_ff=4 * 768, num_heads=12)
    config_gpt2_medium = model_config(num_layers=24, d_model=1024, d_ff=4 * 1024, num_heads=16)
    config_gpt2_large = model_config(num_layers=36, d_model=1280, d_ff=4 * 1280, num_heads=20)
    print("d) GPT-2 scaling FLOPs breakdown, see the picture saved in cs336_basics/resource_analysis/model_flops_breakdown.png")
    draw_model_flops_breakdown(
        ("gpt2_small", config_gpt2_small),
        ("gpt2_medium", config_gpt2_medium),
        ("gpt2_large", config_gpt2_large)
    )
    print("------------------------------------------------------------------------------")
    
    print("e) GPT-2 XL long-context FLOPs analysis")
    analyze_gpt2_xl_long_context_flops(config_gpt2_xl)
    print("------------------------------------------------------------------------------")
    
    print("Summary:"
    "when the context length is relatively small (e.g., 1024 tokens), "
    "the FFN is the dominant FLOPs contributor, taking around 60-70% of total forward pass FLOPs, "
    "while attention takes around 20-30%."
    "However, as we increase the context length to 16,384 tokens, "
    "the attention FLOPs grow quadratically with sequence length, "
    "becoming the dominant contributor at around 80-90% of total FLOPs, "
    "while the FFN share drops to around 10-20%.")
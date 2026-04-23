"""
    Utility Functions for resource analysis
"""
from typing import OrderedDict
import matplotlib.pyplot as plt
from .model_config import model_config
from .flops import model_forward_flops_counting

def print_ordered_result(result: OrderedDict, float_format=".2e"):
    """
        Print result components in a fixed order.
    """
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:{float_format}}")
        else:
            print(f"{key}: {value}")

def draw_model_flops_breakdown(
    *configs: tuple[str, model_config],
    save_path="cs336_basics/resource_analysis/model_flops_breakdown.png",
):
    """
        Draw Flops breakdown picture for all model configs
    """
    model_results = OrderedDict()
    proportion_results = OrderedDict()

    for model_name, config in configs:
        result = model_forward_flops_counting(config)
        model_results[model_name] = result

    if not model_results:
        raise ValueError("model_results are None.")

    total_key = "total_flops"
    component_names = ["attention", "ffn", "output_embedding"]

    for model_name, result in model_results.items():
        total_flops = result[total_key]
        proportion_results[model_name] = OrderedDict()

        for component in component_names:
            flops = result[component]
            proportion_results[model_name][component] = (
                flops / total_flops if total_flops else 0
            )

    model_names = list(proportion_results.keys())
    x_positions = list(range(len(model_names)))
    bar_width = 0.8 / len(component_names)
    center_offset = (len(component_names) - 1) / 2

    plt.figure(figsize=(10, 6))
    for index, component in enumerate(component_names):
        component_values = [
            proportion_results[model_name][component] * 100
            for model_name in model_names
        ]
        plt.bar(
            [x + (index - center_offset) * bar_width for x in x_positions],
            component_values,
            width=bar_width,
            label=component,
        )

    plt.xticks(x_positions, model_names)
    plt.ylabel("FLOPs (% of total forward pass)")
    plt.title("Model Forward FLOPs Breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
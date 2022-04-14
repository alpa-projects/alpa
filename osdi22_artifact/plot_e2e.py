import argparse
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from common import parse_tsv_file, plot_scaling, plot_text, fix_missing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    data_ref = parse_tsv_file("results_e2e_ref.tsv", "e2e")
    data = parse_tsv_file("results_e2e.tsv", "e2e")
    fix_missing(data, data_ref)

    prefix = ""
    suffix = ".pdf"
    num_gpus = [1, 4, 8, 16, 32]
    figure_size = (7, 3)

    if args.model is None:
        plot_models = ["gpt", "moe", "wresne"]
    else:
        plot_models = [args.model]

    if "gpt" in plot_models:
        methods = ["megatron", "parax.inter_only", "parax.intra_only", "parax.auto"]
        plot_scaling(data, "gpt", methods, num_gpus,
                     y_ticks=[0, 1.0, 2.0, 3.0, 4.0],
                     figure_size=figure_size,
                     output=f"{prefix}e2e_gpt{suffix}")

    if "moe" in plot_models:
        methods = ["deepspeed", "parax.inter_only", "parax.intra_only", "parax.auto"]
        plot_scaling(data, "moe", methods, num_gpus,
                     y_ticks=[0, 1.0, 2.0, 3.0],
                     figure_size=figure_size,
                     output=f"{prefix}e2e_moe{suffix}")

    if "wresnet" in plot_models:
        methods = ["parax.ppdp", "parax.inter_only", "parax.intra_only", "parax.auto"]
        plot_scaling(data, "wresnet", methods, num_gpus,
                     y_ticks=[0, 0.2, 0.4, 0.6],
                     figure_size=figure_size,
                     output=f"{prefix}e2e_wresnet{suffix}")

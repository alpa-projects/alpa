import argparse
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from common import parse_tsv_file, plot_scaling, plot_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    data_file = "results_e2e.tsv"
    data = parse_tsv_file(data_file, "e2e")

    prefix = ""
    suffix = ".pdf"
    num_gpus = [1, 4, 8, 16, 32]
    figure_size = (7, 3)

    methods = ["megatron", "parax.inter_only", "parax.intra_only", "parax.auto"]
    plot_scaling(data, "gpt", methods, num_gpus,
                 y_ticks=[0, 1.0, 2.0, 3.0, 4.0],
                 figure_size=figure_size,
                 output=f"{prefix}e2e_gpt{suffix}")

    methods = ["deepspeed", "parax.inter_only", "parax.intra_only", "parax.auto"]
    plot_scaling(data, "moe", methods, num_gpus,
                 y_ticks=[0, 1.0, 2.0, 3.0],
                 figure_size=figure_size,
                 output=f"{prefix}e2e_moe{suffix}")

    methods = ["parax.ppdp", "parax.inter_only", "parax.intra_only", "parax.auto"]
    plot_scaling(data, "wresnet", methods, num_gpus,
                 y_ticks=[0, 0.2, 0.4, 0.6],
                 figure_size=figure_size,
                 output=f"{prefix}e2e_wresnet{suffix}")

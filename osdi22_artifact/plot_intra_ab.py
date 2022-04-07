import argparse
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from common import parse_tsv_file, plot_scaling, plot_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    data_file = "results_intra_ablation.tsv"
    data = parse_tsv_file(data_file, "intra-op-ablation")

    prefix = ""
    suffix = ".pdf"
    methods = ["parax.auto_sharding", "parax.data_parallel", "parax.zero_2", "parax.zero_3", "parax.heuristic"]
    num_gpus = [1, 2, 4, 8]
    figure_size = (7, 3.2)

    plot_scaling(data, "gpt", methods, num_gpus,
                 y_ticks=[0.15, 0.30, 0.45, 0.60],
                 legend_ncol=2,
                 figure_size=figure_size,
                 output=f"{prefix}intra_ab_gpt{suffix}")

    plot_scaling(data, "moe", methods, num_gpus,
                 y_ticks=[0.10, 0.20, 0.30, 0.40],
                 legend_ncol=2,
                 figure_size=figure_size,
                 output=f"{prefix}intra_ab_moe{suffix}")

    plot_scaling(data, "wresnet", methods, num_gpus,
                 y_ticks=[0.02, 0.04, 0.06],
                 legend_ncol=2,
                 figure_size=figure_size,
                 output=f"{prefix}intra_ab_wresnet{suffix}")

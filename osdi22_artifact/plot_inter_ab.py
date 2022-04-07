import argparse
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from common import parse_tsv_file, plot_scaling, plot_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    data_file = "results_inter_ablation.tsv"
    data = parse_tsv_file(data_file, "inter_op_ablation")

    prefix = ""
    suffix = ".pdf"
    #methods = ["parax.auto_stage", "parax.equal_mesh", "parax.equal_layer", "parax.equal_eqn"]
    methods = ["parax.auto_stage", "parax.equal_layer", "parax.equal_eqn"]
    num_gpus = [16]


    #outfile = f"{prefix}inter_ab{suffix}"
    #fig, ax = plt.subplots()
    #gs = gridspec.GridSpec(1, 2) #, width_ratios=[2, 1])
    #ax1 = plt.subplot(gs[0])
    #ax2 = plt.subplot(gs[1])

    plot_scaling(data, "gpt", methods, num_gpus,
                 y_min=0.5,
                 y_ticks=[0.5, 1.0, 1.5, 2.0],
                 legend_ncol=2,
                 figure_size=(2, 3.2),
                 draw_legend=False,
                 draw_xlabel="#GPUs",
                 output=f"{prefix}inter_ab_gpt{suffix}")

    num_gpus = [8, 16, 32]
    plot_scaling(data, "wresnet", methods, num_gpus,
                 y_min=0.02,
                 y_ticks=[0.05, 0.10, 0.15, 0.20, 0.25],
                 legend_ncol=2,
                 figure_size=(4, 3.2),
                 draw_legend=False,
                 draw_xlabel="#GPUs",
                 output=f"{prefix}inter_ab_wresnet{suffix}")

    plot_scaling(data, "wresnet", methods, num_gpus,
                 legend_ncol=4,
                 figure_size=(6, 0.3),
                 draw_legend=True,
                 only_legend=True,
                 output=f"{prefix}inter_ab_legend{suffix}")

    #fig.set_size_inches((7, 3.5))
    #fig.savefig(outfile, bbox_inches='tight')
    #print("Output to %s ..." % outfile)

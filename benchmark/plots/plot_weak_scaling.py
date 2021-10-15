import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


########## Color
method_color_dict = {
    "parax.auto_sharding" : "C0",
    "parax.zero_2"        : "C1",
    "parax.data_parallel" : "C2",

    "OOM": "gray",
}

def method2color(method):
    return method_color_dict[method]


########## Order
method_order_list = [
    "parax.data_parallel",
    "parax.zero_2",
    "parax.auto_sharding",
]

def method2order(method):
    return method_order_list.index(method)


########## Name
show_name_dict = {
    "parax.auto_sharding" : "auto-sharding (ours)",
    "parax.data_parallel" : "data-parallel",
    "parax.zero_2"        : "zero-2",
    "gpt"                 : "GPT",
    "moe"                 : "MoE",
}

def show_name(name):
    return show_name_dict.get(name, name)



def read_raw_data(filename):
    # data[network][num_gpus][method] = {"TFLOPS": xxx}
    data = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda: -1))))

    # read raw data
    for line in open(filename):
        line = line.strip()
        if line.startswith("#") or len(line) < 1:
            continue

        exp_name, instance, num_nodes, num_gpu_per_node, network_name, method, value, time_stamp = \
            line.split('\t')

        if exp_name != "weak_scaling_model":
            continue

        num_gpus = int(num_nodes) * int(num_gpu_per_node)
        data[network_name][num_gpus][method] = eval(value)

    return data


def plot_scaling(raw_data, suffix):
    networks = ["GPT", "W-ResNet", "MoE"]
    methods = ["parax.auto_sharding", "parax.data_parallel", "parax.zero_2"]
    num_gpus = [1, 2, 4, 8]

    # Parameters of the figure
    x0 = 0
    gap = 1.5
    width = 1
    fontsize = 19
    figure_size = (8, 4)
    xticks_font_size = fontsize - 2
    draw_ylabel = "Throughput (TFLOPS)"
    draw_xlabel = "The Number of GPUs"
    legend_bbox_to_anchor = (0.45, 1.25)
    legend_nrow = 1

    methods.sort(key=lambda x: method2order(x))

    for network in networks:
        # Draw one figure for one network
        fig, ax = plt.subplots()

        xticks = []
        xlabels = []
        all_methods = set()
        legend_set = dict()

        for num_gpu in num_gpus:
            ys = []
            colors = []
            hatches = []

            for method in methods:
                if method not in raw_data[network][num_gpu]:
                    continue

                value = raw_data[network][num_gpu][method]["tflops"]
                if value < 0:
                    ys.append(base_y_max / 10)
                    colors.append(method2color("OOM"))
                    hatches.append("/////")
                else:
                    ys.append(value)
                    colors.append(method2color(method))
                    hatches.append(None)

            if num_gpu == 1:
                base_y_max = max(ys)

            # Draw normal bars
            xs = np.arange(x0, x0 + len(ys))
            bars = ax.bar(xs, ys, width=width, color=colors)
            for method, bar_obj in zip(methods, bars):
                all_methods.add(method)
                if method not in legend_set:
                    legend_set[method] = bar_obj

            # Draw OOM bars
            for i, (hatch, bar) in enumerate(zip(hatches, bars)):
                if hatch:
                    bar.set_edgecolor('white')
                    bar.set_hatch(hatch)
                    ax.text(xs[i], ys[i], "OOM",
                            horizontalalignment='center', verticalalignment='bottom',
                            fontsize=fontsize // 2)

            x0 += len(ys) + gap
            xticks.append(x0 - gap - len(ys)*width/2.0 - width/2.0)
            xlabels.append(str(num_gpu))

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=xticks_font_size)
        ax.set_xlabel(draw_xlabel, fontsize=fontsize)

        ax.set_yticks(ax.get_yticks().tolist())
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.set_ylabel(draw_ylabel, fontsize=fontsize)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(linewidth=0.5, linestyle="dotted") # draw grid line
        ax.set_axisbelow(True)  # grid lines are behind the rest
        ax.tick_params(bottom=False, top=False, right=False)

        ax.axhline(base_y_max, 0, x0, color="black", linestyle="dotted")

        all_methods = list(all_methods)
        all_methods.sort(key=lambda x : method2order(x))

        # Put the legend outside the plot
        ncol = (len(all_methods) + legend_nrow - 1)// legend_nrow
        ax.legend([legend_set[x] for x in all_methods],
                  [show_name(x) for x in all_methods],
                  fontsize=fontsize-1,
                  loc='upper center',
                  bbox_to_anchor=legend_bbox_to_anchor,
                  ncol=ncol,
                  handlelength=1.0,
                  handletextpad=0.5,
                  columnspacing=1.1)

        # Save the figure
        if args.show:
            plt.show()
            return
        else:
            output = f"weak-scaling-{network}.{suffix}"
            fig.set_size_inches(figure_size)
            fig.savefig(output, bbox_inches='tight')
            print("Output the plot to %s" % output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    filename = "result_weak_scaling.tsv"
    suffix = "png"
    raw_data = read_raw_data(filename)

    plot_scaling(raw_data, suffix)

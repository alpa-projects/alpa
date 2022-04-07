import argparse
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['hatch.linewidth'] = 0.5

########## Color
method_color_dict = {
    # e2e
    "megatron"            : "C3",
    "deepspeed"           : "C5",
    "parax.ppdp"          : "C4",
    "parax.intra_only"    : "C2",
    "parax.inter_only"    : "C1",
    "parax.auto"          : "C0",

    # intra op ablation
    "parax.data_parallel" : "C5",
    "parax.heuristic"     : "C3",
    "parax.zero_2"        : "C1",
    "parax.zero_3"        : "C2",
    "parax.auto_sharding" : "C0",

    # inter op ablation
    "parax.equal_eqn"     : "C2",
    "parax.equal_layer"   : "C1",
    "parax.equal_mesh"    : "C1",
    "parax.auto_stage"    : "C0",

    # cross mesh resharding
    "parax.signal_send_recv"        : "C2",
    "parax.without_local_allgather" : "C1",
    "parax.with_local_allgather"    : "C0",
}


def method2color(method):
    return method_color_dict[method]


method_hatch_dict = {
    # e2e
    "megatron"            : "x",
    "deepspeed"           : "x",
    "parax.ppdp"          : "x",
    "parax.inter_only"    : ".",
    "parax.intra_only"    : "+",
    "parax.auto"          : "",

    # intra op ablation
    "parax.data_parallel" : "x",
    "parax.heuristic"     : ".",
    "parax.zero_2"        : "|",
    "parax.zero_3"        : "-",
    "parax.auto_sharding" : "",

    # inter op ablation
    "parax.equal_eqn"     : "x",
    "parax.equal_layer"   : "|",
    "parax.equal_mesh"    : ".",
    "parax.auto_stage"    : "",

    # cross mesh resharding
    "parax.signal_send_recv"        : "x",
    "parax.without_local_allgather" : ".",
    "parax.with_local_allgather"    : "",
}

def method2hatch(method):
    return method_hatch_dict[method]


########## Order
method_order_list = [
    # e2e
    "megatron",
    "deepspeed",
    "parax.ppdp",
    "parax.inter_only",
    "parax.intra_only",
    "parax.manual",
    "parax.auto",

    # intra op ablation
    "parax.data_parallel",
    "parax.heuristic",
    "parax.zero_2",
    "parax.zero_3",
    "parax.auto_sharding",

    # inter op ablation
    "parax.equal_eqn",
    "parax.equal_layer",
    "parax.equal_mesh",
    "parax.auto_stage",

    # cross mesh resharding
    "parax.signal_send_recv",
    "parax.without_local_allgather",
    "parax.with_local_allgather",

    "Linear-scaling",
]

def method2order(method):
    return method_order_list.index(method)

########## Rename
show_name_dict = {
    # e2e
    "megatron"            : "Megatron-LM",
    "deepspeed"           : "Deepspeed",
    "parax.ppdp"          : "PP-DP",
    "parax.inter_only"    : "Inter-op only",
    "parax.intra_only"    : "Intra-op only",
    "parax.manual"        : "Alpa (manual)",
    "parax.auto"          : "Alpa (ours)",

    # intra op ablation
    "parax.auto_sharding" : "ILP (ours)",
    "parax.data_parallel" : "Data",
    "parax.zero_2"        : "ZeRO-2",
    "parax.zero_3"        : "ZeRO-3",
    "parax.heuristic"     : "Heuristic",

    # inter op ablation
    "parax.auto_stage"    : "DP (ours)",
    "parax.equal_eqn"     : "Equal operator",
    "parax.equal_mesh"    : "Equal mesh",
    "parax.equal_layer"   : "Equal layer",
    "parax.heuristic"     : "Heuristic",

    # cross mesh resharding
    "parax.signal_send_recv"        : "Signal send/recv",
    "parax.without_local_allgather" : "w/o local all-gather",
    "parax.with_local_allgather"    : "w/ local all-gather",

    "gpt"                 : "GPT",
    "moe"                 : "MoE",
}

def show_name(name):
    return show_name_dict.get(name, name)


def enhance_color(color, h=1, l=1, s=1):
    """Make color looks better for pyplot"""
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))

    h, l, s = h * c[0], l * c[1], s * c[2]
    h, l, s = [max(min(x, 1), 0) for x in [h, l, s]]

    return colorsys.hls_to_rgb(h, l, s)


def parse_tsv_file(filename, wanted_exp_name):
    # Output format.
    # data[network][num_gpus][method] = {"tflops": 43.2, "latencies": [0.22, 0.23, 0.22]}
    data = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda: -1))))

    # Parse string
    for line in open(filename):
        line = line.strip()
        if line.startswith("#") or len(line) < 1:
            continue

        exp_name, instance, num_nodes, num_gpus_per_node, network_name, method, value, time_stamp = \
            line.split('\t')

        if exp_name != wanted_exp_name:
            continue

        num_gpus = int(num_nodes) * int(num_gpus_per_node)
        value = eval(value)
        if method not in data[network_name][num_gpus]:
           data[network_name][num_gpus][method] = value
        elif value["tflops"] > data[network_name][num_gpus][method]["tflops"]:
           # pick the best
           data[network_name][num_gpus][method] = value


    return data


def plot_scaling(raw_data, network, methods, num_gpus,
                 figax=None,
                 y_ticks=None, y_min=0,
                 draw_ylabel="Throughput (PFLOPS)",
                 draw_xlabel="The number of GPUs",
                 draw_legend=True, only_legend=False,
                 legend_ncol=1,
                 figure_size=(8, 4), output="out.png"):
    # Parameters of the figure
    x0 = 0
    gap = 1.5
    width = 1
    fontsize = 19
    xticks_font_size = fontsize - 2

    methods.sort(key=lambda x: method2order(x))

    if figax is None:
        fig, ax = plt.subplots()
        axes = []
        axes.append(ax)
    else:
        # for drawing subplot
        ax = figax

    # Draw one figure for one network
    xticks = []
    xlabels = []
    all_methods = set()
    legend_set = dict()
    base_y_max = None

    for num_gpu in num_gpus:
        ys = []
        colors = []
        hatches = []

        bar_methods = []
        for method in methods:
            if method not in raw_data[network][num_gpu]:
                print(f"Data for (network, num_gpu, method)={(network, num_gpu, method)} is missing")
                continue

            bar_methods.append(method)
            value = raw_data[network][num_gpu][method]["tflops"] / 1e3
            ys.append(num_gpu * max(value, 0))
            colors.append(method2color(method))
            hatches.append(method2hatch(method))

        if num_gpu == 1:
            base_y_max = max(ys)

        # Draw bars
        xs = np.arange(x0, x0 + len(ys))
        bars = ax.bar(xs, ys, width=width, color=colors, hatch=hatches, edgecolor="white")
        for method, bar_obj in zip(bar_methods, bars):
            all_methods.add(method)
            if method not in legend_set:
                legend_set[method] = bar_obj

        # Draw linear scaling bar
        if base_y_max is not None:
            bars = ax.bar(np.mean(xs), base_y_max * num_gpu,
                          width=width*len(bars),
                          color='none', lw=1, edgecolor="black")
            legend_set["Linear-scaling"] = bars

        # Draw OOM text
        for i in range(len(xs)):
            if ys[i] <= 1e-9:
                ax.text(xs[i], 0, "X", color=colors[i],
                        horizontalalignment='center', verticalalignment='bottom',
                        fontsize=fontsize * 0.8)

        x0 += len(ys) + gap
        xticks.append(x0 - gap - len(ys)*width/2.0 - width/2.0)
        xlabels.append(str(num_gpu))

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=xticks_font_size)
    ax.set_xlabel(draw_xlabel, fontsize=fontsize + 2)

    if y_ticks:
        ax.set_yticks(y_ticks)
    else:
        ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
    ax.set_ylabel(draw_ylabel, fontsize=fontsize - 1)
    ax.set_ylim(bottom=y_min)
    ax.yaxis.grid(linewidth=0.5, linestyle="dotted") # draw grid line
    ax.set_axisbelow(True)  # grid lines are behind the rest
    ax.tick_params(bottom=False, top=False, right=False)

    all_methods = list(all_methods) + (["Linear-scaling"] if base_y_max is not None else [])
    all_methods.sort(key=lambda x : method2order(x))

    # Draw legend
    if draw_legend:
        if only_legend:
            fig, ax = plt.subplots()
            ax.axis(False)

        ax.legend([legend_set[x] for x in all_methods],
                  [show_name(x) for x in all_methods],
                  ncol=legend_ncol,
                  fontsize=fontsize-2,
                  loc='upper left',
                  handlelength=1.0,
                  handletextpad=0.5,
                  columnspacing=1.1)

    # Save the figure
    if figax is None:
        fig.set_size_inches(figure_size)
        fig.savefig(output, bbox_inches='tight')
        print("Output the plot to %s" % output)


def plot_text(ax, x, y, text, rotation=0):
    ax.text(x, y, text, horizontalalignment='center',
            rotation=rotation,
            verticalalignment='center', transform=ax.transAxes,
            fontsize=ax.yaxis.label.get_size())

def plot_compile(data, num_gpus,
                 figax=None,
                 y_ticks=None, draw_ylabel="Time (s)",
                 draw_xlabel="The number of GPUs",
                 figure_size=(8, 4), output="out.png"):
    # Parameters of the figure
    x0 = 0
    gap = 1.5
    width = 1
    fontsize = 19
    xticks_font_size = fontsize - 2

    if figax is None:
        fig, ax = plt.subplots()
        axes = []
        axes.append(ax)
    else:
        # for drawing subplot
        ax = figax

    # Draw one figure for one network
    xticks = []
    xlabels = []
    all_methods = set()
    legend_set = dict()
    base_y_max = None

    for i, num_gpu in enumerate(num_gpus):
        ys = []

        bar_methods = []
        value = data[i]
        ys.append(value)

        # Draw bars
        xs = np.arange(x0, x0 + len(ys))
        bars = ax.bar(xs, ys, width=width, color="C0", edgecolor="white")

        x0 += len(ys) + gap
        xticks.append(x0 - gap - len(ys)*width/2.0 - width/2.0)
        xlabels.append(str(num_gpu))

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=xticks_font_size)
    ax.set_xlabel(draw_xlabel, fontsize=fontsize + 2)

    if y_ticks:
        ax.set_yticks(y_ticks)
    else:
        ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
    ax.set_ylabel(draw_ylabel, fontsize=fontsize - 1)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(linewidth=0.5, linestyle="dotted") # draw grid line
    ax.set_axisbelow(True)  # grid lines are behind the rest
    ax.tick_params(bottom=False, top=False, right=False)


    # Save the figure
    if figax is None:
        fig.set_size_inches(figure_size)
        fig.savefig(output, bbox_inches='tight')
        print("Output the plot to %s" % output)

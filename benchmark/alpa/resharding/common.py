from collections import defaultdict
import math
from typing import Dict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['hatch.linewidth'] = 0.5

########## Color
method_color_dict = {
    # microbenchmark
    "send-recv": "C4",
    "alpa": "C3",
    "ours": "C0",

    # e2e
    # send-recv: C4, alpa: C3, ours: C0
    "ours.broadcast": "C1",

    # ablation single overlap direction
    # alpa: C3
    "ours.overlap-forward": "C2",
    "ours.overlap-backward": "C3",
    "ours.overlap-both": "C0",

    # ablation load-balance
    "ours.balance-size": "C2",
    "ours.no-balance": "C1",
    "ours.balance-both": "C0",

    #
    "signal.send-recv": "C5",
}


def method2color(method):
    return method_color_dict[method]


method_hatch_dict = {
    # microbenchmark
    "send-recv": "x",
    "alpa": "|",
    "ours": "",

    # e2e
    # send-recv: x, alpa: |, ours: .
    "ours.broadcast": "-",

    # ablation single overlap direction
    # alpa: |
    "ours.overlap-forward": ".",
    "ours.overlap-backward": "x",
    "ours.overlap-both": "",

    # ablation load-balance
    "ours.balance-size": ".",
    "ours.no-balance": "-",
    "ours.balance-both": "",

    #
    "signal.send-recv": "+",
}


def method2hatch(method):
    return method_hatch_dict[method]


########## Order
method_order_list = [
    # microbenchmark
    "send-recv",
    "alpa",

    # e2e
    "ours.broadcast",
    "ours",

    # ablation single overlap direction
    # alpa
    "ours.overlap-forward",
    "ours.overlap-backward",
    "ours.overlap-both",

    # ablation load-balance
    "ours.no-balance",
    "ours.balance-size",
    "ours.balance-both",

    #
    "signal.send-recv",
]


def method2order(method):
    return method_order_list.index(method)


########## Rename
show_name_dict = {
    # microbenchmark
    "send-recv": "Send/Recv",
    "alpa": "Alpa",
    "ours": "Ours",

    # e2e
    # send-recv: x, alpa: |, ours: .
    "ours.broadcast": "Broadcast",

    # ablation single overlap direction
    # alpa: |
    "ours.broadcast": "Broadcast",
    "ours.overlap-backward": "Overlap",
    "ours.overlap-both": "Eager-1F1B",

    # ablation load-balance
    "ours.no-balance": "Naive",
    "ours.balance-size": "Load Balance Only",
    "ours.balance-both": "Ours",

    #
    "signal.send-recv": "Signal Send/Recv",
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


########## Parse
# The raw data's format:
# end-to-end
# rd[exp_group][exp_setup][method_setup] = {"tflops": (val, std), "memory": val}
# microbenchmark
# rd[exp_group][exp_setup][method_setup] = {"latency": (val, std)}
def parse_tsv_file(filename):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: (-1, -1)))))

    # Parse string
    for line in open(filename):
        line = line.strip()
        if line.startswith("#") or len(line) < 1:
            continue

        (exp_group_name, exp_setup, method, num_hosts, num_devices_per_host,
         value) = line.split('\t')
        num_gpu = int(num_hosts) * int(num_devices_per_host)

        value = eval(value)
        results = data[exp_group_name][exp_setup]
        if "tflops" in value:
            value["tflops"] = tuple(v * num_gpu for v in value["tflops"])
        if method not in results:
            results[method] = value

    return data


########## Plot
def plot_benchmark_case(raw_data: Dict[str, Dict[str, float]],
                        y_name,
                        methods,
                        figax=None,
                        y_ticks=None,
                        y_min=0,
                        process_y_fn=None,
                        draw_ylabel="Throughput (TFLOPS)",
                        draw_xlabel="",
                        draw_legend=True,
                        only_legend=False,
                        draw_err=True,
                        legend_ncol=1,
                        figure_size=(8, 4),
                        output="out.png"):
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

    xticks = []
    xlabels = []
    all_methods = set()
    legend_set = dict()

    for experiment_setup in raw_data:
        ys = []
        yerrs = []
        colors = []
        hatches = []
        group_results = raw_data[experiment_setup]
        bar_methods = []
        for method in methods:
            if method not in group_results:
                print(
                    f"Data for experiment {experiment_setup}, method {method} is missing"
                )
                continue
            bar_methods.append(method)
            result = group_results[method][y_name]
            if y_name == "memory":
                value = result
            else:
                value, variance = result
                err = math.sqrt(variance)
                if process_y_fn:
                    err = process_y_fn(err)
                yerrs.append(err)
            if process_y_fn:
                value = process_y_fn(value)
            ys.append(value)
            colors.append(method2color(method))
            hatches.append(method2hatch(method))

        # Draw bars
        xs = np.arange(x0, x0 + len(ys))
        bars = ax.bar(xs,
                      ys,
                      width=width,
                      color=colors,
                      hatch=hatches,
                      edgecolor="white")
        if draw_err:
            ax.errorbar(xs, ys, yerrs)
        for method, bar_obj in zip(bar_methods, bars):
            all_methods.add(method)
            if method not in legend_set:
                legend_set[method] = bar_obj

        # Add current x
        x0 += len(ys) + gap
        xticks.append(x0 - gap - len(ys) * width / 2.0 - width / 2.0)
        xlabels.append(experiment_setup)

    # Set tick and labels
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
    ax.yaxis.grid(linewidth=0.5, linestyle="dotted")  # draw grid line
    ax.set_axisbelow(True)  # grid lines are behind the rest
    ax.tick_params(bottom=False, top=False, right=False)

    all_methods = list(all_methods)
    all_methods.sort(key=lambda x: method2order(x))

    # Draw legend
    if draw_legend:
        if only_legend:
            fig, ax = plt.subplots()
            ax.axis(False)

        ax.legend([legend_set[x] for x in all_methods],
                  [show_name(x) for x in all_methods],
                  ncol=legend_ncol,
                  fontsize=fontsize - 4,
                  loc='upper left',
                  handlelength=1.0,
                  handletextpad=0.5,
                  columnspacing=1)

    # Save the figure
    if figax is None:
        fig.set_size_inches(figure_size)
        fig.savefig(output, bbox_inches='tight')
        print("Output the plot to %s" % output)

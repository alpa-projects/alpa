import argparse

from common import parse_tsv_file, plot_benchmark_case

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    data_file = "results_microbenchmark_single_node.tsv"

    prefix = "./eval_"
    suffix = ".pdf"

    methods = ["send-recv", "alpa", "ours"]

    data = parse_tsv_file(data_file)
    y_ticks = [0, 1000, 2000, 3000]
    plot_benchmark_case(data,
                        "latency",
                        methods,
                        y_ticks=y_ticks,
                        draw_ylabel="latency(ms)",
                        draw_xlabel="num GPUs",
                        draw_legend=False,
                        draw_err=False,
                        figure_size=(7, 3),
                        output=f"{prefix}microbenchmark_single_node{suffix}")
    plot_benchmark_case(data,
                        "latency",
                        methods,
                        y_ticks=y_ticks,
                        draw_legend=True,
                        only_legend=True,
                        legend_ncol=3,
                        figure_size=(5, 0.3),
                        output=f"{prefix}microbenchmark_legend{suffix}")
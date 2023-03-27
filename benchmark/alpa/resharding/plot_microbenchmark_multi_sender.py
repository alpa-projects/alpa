import argparse

from common import parse_tsv_file, plot_benchmark_case

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # global config
    prefix = "./eval_"
    suffix = ".pdf"
    # case specialized config
    methods = ["send-recv", "alpa", "ours"]
    y_ticks = [0, 10, 20, 30]
    # load data
    data_file = "results.tsv"
    data = parse_tsv_file(data_file)
    # draw graph
    plot_benchmark_case(data["microbenchmark.multi-sender"],
                        "latency",
                        methods,
                        y_ticks=y_ticks,
                        process_y_fn=lambda x: 16 / (x / 1000),
                        draw_ylabel="Effective Bandwidth(Gbps)",
                        draw_legend=True,
                        legend_ncol=3,
                        draw_err=False,
                        figure_size=(11, 3),
                        output=f"{prefix}microbenchmark_multi_sender{suffix}")

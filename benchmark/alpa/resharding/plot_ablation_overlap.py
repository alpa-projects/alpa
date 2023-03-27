import argparse

from common import parse_tsv_file, plot_benchmark_case

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # global config
    prefix = "./eval_"
    suffix = ".pdf"
    # case specialized config
    methods = ["ours.broadcast", "ours.overlap-backward", "ours.overlap-both"]
    y_ticks = [20, 40, 60]
    # load data
    data_file = "results.tsv"
    data = parse_tsv_file(data_file)
    # draw graph
    plot_benchmark_case(data["ablation.overlap"],
                        "tflops",
                        methods,
                        y_ticks=y_ticks,
                        y_min=20,
                        draw_ylabel="Throughput(TFLOPS)",
                        draw_xlabel="#microbatches",
                        draw_legend=True,
                        draw_err=False,
                        figure_size=(7.5, 3),
                        legend_ncol=2,
                        output=f"{prefix}ablation_overlap{suffix}")

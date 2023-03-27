import argparse

from common import parse_tsv_file, plot_benchmark_case

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # global config
    prefix = "./eval_"
    # case specialized config
    methods = ["send-recv", "alpa", "ours"]
    size_to_name = {1: "1MB", 16: "16MB", 64: "64MB", 1024: "1GB"}
    # load data
    data_file = f"results.tsv"
    data = parse_tsv_file(data_file)
    # draw graph
    for data_size in [1, 16, 64, 1024]:
        y_ticks = [0, 4, 8, 12]
        suffix = f"_{data_size}.pdf"
        size_name = size_to_name[data_size]

        # 1024: M -> G, 32: int32 -> bit
        data_size_gb = data_size / 1024 * 8
        plot_benchmark_case(
            data[f"microbenchmark.single-node.{size_name}"],
            "latency",
            methods,
            y_ticks=y_ticks,
            process_y_fn=lambda x: data_size_gb / (x / 1000),
            draw_ylabel="Effective Bandwidth(Gbps)",
            draw_xlabel="#GPU",
            draw_legend=False,
            draw_err=False,
            figure_size=(7, 3),
            output=f"{prefix}microbenchmark_single_node{suffix}")

        y_ticks = [0, 4, 8, 12]
        plot_benchmark_case(
            data[f"microbenchmark.multi-node.{size_name}"],
            "latency",
            methods,
            y_ticks=y_ticks,
            process_y_fn=lambda x: data_size_gb / (x / 1000),
            draw_ylabel="Effective Bandwidth(Gbps)",
            draw_xlabel="#host",
            draw_legend=False,
            draw_err=False,
            figure_size=(7, 3),
            output=f"{prefix}microbenchmark_multi_node{suffix}")

    suffix = ".pdf"
    plot_benchmark_case(data["microbenchmark.multi-node.1GB"],
                        "latency",
                        methods,
                        y_ticks=y_ticks,
                        draw_legend=True,
                        only_legend=True,
                        legend_ncol=3,
                        figure_size=(5, 0.3),
                        output=f"{prefix}microbenchmark_legend{suffix}")

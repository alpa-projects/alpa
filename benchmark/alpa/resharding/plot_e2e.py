import argparse

from common import parse_tsv_file, plot_benchmark_case

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # global config
    prefix = "./eval_"
    suffix = ".pdf"
    # case specialized config
    methods = ["send-recv", "alpa", "ours.broadcast", "ours", "signal.send-recv"]
    yticks = {"gpt": [200, 300, 400, 500], "unet": [20, 40, 60]}
    figure_size = {"gpt": (7, 3), "unet": (2.5, 3)}
    # load data
    data_file = "results.tsv"
    data = parse_tsv_file(data_file)
    # draw graph
    for model in ["gpt", "unet"]:
        plot_benchmark_case(data[f"e2e.{model}"],
                            "tflops",
                            methods,
                            y_ticks=yticks[model],
                            y_min=yticks[model][0],
                            draw_legend=False,
                            draw_err=False,
                            figure_size=figure_size[model],
                            output=f"{prefix}e2e_{model}{suffix}")

    plot_benchmark_case(data["e2e.gpt"],
                        "tflops",
                        methods,
                        legend_ncol=len(methods),
                        only_legend=True,
                        figure_size=(9, 0.3),
                        output=f"{prefix}e2e_legend{suffix}")

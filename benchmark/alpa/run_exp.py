"""Convenient script to run search experiments with mutliple cluster
settings."""
import os
import sys
import argparse
from datetime import datetime

from benchmark import benchmark_suite


def run_exp(cluster_settings, suite_name, benchmark_settings=None):
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    with open(f"{now}-{suite_name}.log", "w") as log_file:
        # re-open stdout without buffering
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

        # redirect stdout and stderr to the log file opened above
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())

        if benchmark_settings is None:
            benchmark_settings = {}
        for num_hosts, num_devices_per_host in cluster_settings:
            num_gpus = num_hosts * num_devices_per_host
            benchmark_suite(suite_name, num_hosts, num_devices_per_host,
                            exp_name=f"{suite_name}_{num_gpus}_gpus",
                            disable_tqdm=True, **benchmark_settings)

model_search_suites = {
    "gpt": ("gpt.grid_search_auto", {}),
    "moe": ("moe.grid_search_auto", {}),
    "wresnet": ("wresnet.grid_search_auto", {}),
    "gpt_inference": ("gpt_inference.profile", {}),
    "gpt_no_embedding_inference": ("gpt_no_embedding_inference.profile", {}),
    "gpt_inference_streaming": ("gpt_inference.profile", {"profile_driver_time": True}),
}
cluster_settings = [(8, 8), (4, 8), (2, 8), (1, 8), (1, 4), (1, 2), (1, 1)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=model_search_suites.keys())
    args = parser.parse_args()
    run_exp(cluster_settings, model_search_suites[args.model])

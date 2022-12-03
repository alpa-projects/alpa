"""The entry point of intra-op + inter-op parallelism benchmark."""
import os
import argparse
import time
import multiprocessing as mp
import json

import numpy as np
from alpa.util import (write_tsv, get_num_hosts_and_num_devices, to_str_round,
                       GB)
from alpa import init
from alpa.util import disable_tqdm_globally

from benchmark_cross_mesh_resharding import benchmark_one_case_internal
import suite


benchmark_suites = {
    "n-to-m": suite.perf_n_to_m_suite,
    "1-to-m": suite.perf_1_to_m_suite,
}

def benchmark_and_write_to_namespace(result_namespace, *args, **kwargs):
    result = benchmark_one_case_internal(*args, **kwargs)
    result_namespace.result = result


def benchmark_one_case(*args, use_separate_process=False, **kwargs):
    if not use_separate_process:
        return benchmark_one_case_internal(*args, **kwargs)
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    result_namespace = manager.Namespace()
    p = ctx.Process(target=benchmark_and_write_to_namespace,
                    args=(result_namespace, *args),
                    kwargs=kwargs)
    p.start()
    p.join()
    if p.exitcode != 0:
        return -1, -1, [-1], -1, None
    return result_namespace.result

def benchmark_load_balance_suite():
    os.makedirs("tmp", exist_ok=True)
    
    result_file = "tmp/n_to_m_result.json"
    result = []

    suite = benchmark_suites["n-to-m"]
    config_list = [
        {"resharding_mode": "send_recv", "resharding_loadbalance_mode": "normal", "use_local_allgather":False},
        {"resharding_mode": "send_recv", "resharding_loadbalance_mode": "normal", "use_local_allgather":True},
        {"resharding_mode": "broadcast", "resharding_loadbalance_mode": "no_loadbalance", "use_local_allgather":False},
        {"resharding_mode": "broadcast", "resharding_loadbalance_mode": "loadbalance_size", "use_local_allgather":False},
        {"resharding_mode": "broadcast", "resharding_loadbalance_mode": "loadbalance_order", "use_local_allgather":False},
    ]

    # Run all cases
    for key, benchmark_case in suite.items():
        # Run one case
        for config in config_list:
            print("Working on {}: {}, config: {}".format(key, str(benchmark_case), str(config)))
            one_result = benchmark_one_case(benchmark_case.src_mesh_shape,
                                        benchmark_case.dst_mesh_shape,
                                        benchmark_case.src_sharding_spec,
                                        benchmark_case.dst_sharding_spec,
                                        benchmark_case.tensor_shape,
                                        config["resharding_mode"],
                                        config["use_local_allgather"],
                                        config["resharding_loadbalance_mode"])

            print(one_result)
            result.append(one_result)
            json.dump(result, open(result_file, "w"))
            
            time.sleep(0.1)  # for ctrl+c to work

def benchmark_1_to_m_suite():
    os.makedirs("tmp", exist_ok=True)

    result_file = "tmp/1_to_m_result.json"
    result = []

    suite = benchmark_suites["1-to-m"]
    config_list = [
        {"resharding_mode": "send_recv", "resharding_loadbalance_mode": "normal", "use_local_allgather":False},
        {"resharding_mode": "send_recv", "resharding_loadbalance_mode": "normal", "use_local_allgather":True},
        {"resharding_mode": "broadcast", "resharding_loadbalance_mode": "normal", "use_local_allgather":False},
    ]

    # Run all cases
    for key, benchmark_case in suite.items():
        # Run one case
        for config in config_list:
            print("Working on {}: {}, config: {}".format(key, str(benchmark_case), str(config)))
            one_result = benchmark_one_case(benchmark_case.src_mesh_shape,
                                        benchmark_case.dst_mesh_shape,
                                        benchmark_case.src_sharding_spec,
                                        benchmark_case.dst_sharding_spec,
                                        benchmark_case.tensor_shape,
                                        config["resharding_mode"],
                                        config["use_local_allgather"],
                                        config["resharding_loadbalance_mode"])
            print(one_result)
            result.append(one_result)
            json.dump(result, open(result_file, "w"))

            time.sleep(0.1)  # for ctrl+c to work



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",
                        choices=["1-to-m", "load-balance"],
                        type=str,
                        required=True)
    args = parser.parse_args()

    if args.suite == "1-to-m":
        benchmark_1_to_m_suite()
    else:
        benchmark_load_balance_suite()

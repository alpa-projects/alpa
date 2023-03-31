"""The entry point of intra-op + inter-op parallelism benchmark."""
import argparse
import json
import multiprocessing as mp
import os
import time

from benchmark_cross_mesh_resharding import benchmark_one_case_internal
import suite


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


def benchmark_n_to_m_suite(functional_test):
    os.makedirs("tmp", exist_ok=True)

    result_file = "tmp/n_to_m_result.json"
    result = []

    benchmark_cases = suite.perf_n_to_m_suite
    resharding_config_list = suite.resharding_n_to_m_configs

    # Run all cases
    for case_name, benchmark_case in benchmark_cases.items():
        # Run one case
        for config in resharding_config_list:
            print("Working on {}: {}, config: {}".format(
                case_name, str(benchmark_case), str(config)))
            one_result = benchmark_one_case(
                benchmark_case.src_mesh_shape, benchmark_case.dst_mesh_shape,
                benchmark_case.src_sharding_spec,
                benchmark_case.dst_sharding_spec, benchmark_case.tensor_shape,
                config["resharding_mode"], config["use_local_allgather"],
                config["resharding_loadbalance_mode"], functional_test)

            print(one_result)
            result.append(one_result)
            json.dump(result, open(result_file, "w"), indent=4)

            time.sleep(0.1)  # for ctrl+c to work


def benchmark_1_to_m_suite(functional_test):
    os.makedirs("tmp", exist_ok=True)

    result_file = "tmp/1_to_m_result.json"
    result = []

    benchmark_cases = suite.perf_1_to_m_suite
    resharding_config_list = suite.resharding_1_to_m_configs

    # Run all cases
    for case_name, benchmark_case in benchmark_cases.items():
        # Run one case
        for config in resharding_config_list:
            print("Working on {}: {}, config: {}".format(
                case_name, str(benchmark_case), str(config)))
            one_result = benchmark_one_case(
                benchmark_case.src_mesh_shape, benchmark_case.dst_mesh_shape,
                benchmark_case.src_sharding_spec,
                benchmark_case.dst_sharding_spec, benchmark_case.tensor_shape,
                config["resharding_mode"], config["use_local_allgather"],
                config["resharding_loadbalance_mode"], functional_test)
            print(one_result)
            result.append(one_result)
            json.dump(result, open(result_file, "w"), indent=4)

            time.sleep(0.1)  # for ctrl+c to work


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",
                        choices=["1-to-m", "n-to-m"],
                        type=str,
                        required=True)
    parser.add_argument("--functional-test", action="store_true")
    args = parser.parse_args()

    if args.suite == "1-to-m":
        benchmark_1_to_m_suite(args.functional_test)
    else:
        benchmark_n_to_m_suite(args.functional_test)

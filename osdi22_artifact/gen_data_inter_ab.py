"""The entry point of intra-op + inter-op parallelism benchmark."""
import argparse
from datetime import datetime
import time

import numpy as np

from alpa.util import write_tsv, run_cmd, get_num_hosts_and_num_devices, to_str_round, GB

from benchmark.alpa.benchmark_3d_one_case import benchmark_one_case
from benchmark.alpa.suite_paper_wresnet import paper_ablation_wresnet_suite
from suite_artifact_e2e_gpt import artifact_search_e2e_gpt_suite

benchmark_suites = {
    "gpt.inter_op": artifact_search_e2e_gpt_suite,
    "wresnet.inter_op": paper_ablation_wresnet_suite,
}


def run_equal_eqn_one_case(model, case, niter, num_hosts, num_devices_per_host,
                           use_separate_process, disable_tqdm):
    ablation_config = {"use_equal_eqn": True}
    case[-1]["use_hlo_cost_model"] = False
    return benchmark_one_case(model, case, niter, num_hosts,
                              num_devices_per_host, use_separate_process, True,
                              disable_tqdm, ablation_config)


def run_equal_layer_one_case(model, case, niter, num_hosts,
                             num_devices_per_host, use_separate_process,
                             disable_tqdm):
    optimal_result = [0] * 6
    num_stages = 1
    if model == "moe" or model == "gpt":
        num_layers = case[3]
    elif model == "wresnet":
        num_layers = case[2]
    while num_layers % num_stages == 0:
        ablation_config = {"num_stages": num_stages}
        if case[-1]:
            case[-1]["ablation_equal_layer"] = True
        else:
            case[-1] = {"ablation_equal_layer": True}
        result = benchmark_one_case(model, case, niter, num_hosts,
                                    num_devices_per_host, use_separate_process,
                                    True, disable_tqdm, ablation_config)
        if result[5] > optimal_result[5]:
            optimal_result = result
        num_stages *= 2
    return optimal_result


def run_ablation_one_case(model, case, niter, num_hosts, num_devices_per_host,
                          use_separate_process, disable_tqdm):
    results = []
    results.append(
        benchmark_one_case(model, case, niter, num_hosts, num_devices_per_host,
                           use_separate_process, True, disable_tqdm))
    results.append(
        run_equal_layer_one_case(model, case, niter, num_hosts,
                                 num_devices_per_host, use_separate_process,
                                 disable_tqdm))
    results.append(
        run_equal_eqn_one_case(model, case, niter, num_hosts,
                               num_devices_per_host, use_separate_process,
                               disable_tqdm))
    return results


def benchmark_one_suite(suite_name, num_hosts, num_devices_per_host, exp_name,
                        niter, use_separate_process, disable_tqdm):
    num_gpus = num_hosts * num_devices_per_host
    try:
        suite = benchmark_suites[suite_name][num_gpus]
    except KeyError:
        suite = None
    if not suite:
        print(
            f"No available benchmark suite for {suite_name} on {num_gpus} GPUs")
        exit()
    run_cmd("mkdir -p tmp")

    model_type = suite_name.split(".")[0]
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_name = f"{model_type}_alpa_{exp_name}_{date_str}.tsv"

    # Run all cases
    for benchmark_case in suite:
        if model_type in ["gpt"]:
            (batch_size, seq_len, hidden_size, num_layers, num_heads,
             vocab_size, l_dim0, l_dim1, p_dim0, p_dim1, pipeline_mp_size,
             num_micro_batches, force_batch_dim_mapping, use_remat,
             prefer_reduce_scatter, pipeline_stage_mode,
             overwrite_global_config_dict) = benchmark_case
            model_config = (batch_size, seq_len, hidden_size, num_layers,
                            num_heads)
        elif model_type == "wresnet":
            (batch_size, image_size, num_layers, num_channels, width_factor,
             dtype, num_micro_batches, force_batch_dim_mapping,
             prefer_reduce_scatter, use_remat, _) = benchmark_case
            model_config = (batch_size, image_size, num_layers, num_channels,
                            width_factor)
            overwrite_global_config_dict = {}
            pipeline_stage_mode = "auto_gpipe"
            pipeline_mp_size = 1
        else:
            raise ValueError(f"Invalid model: {model_type}")

        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))
        results = run_ablation_one_case(model_type, benchmark_case, niter,
                                        num_hosts, num_devices_per_host,
                                        use_separate_process, disable_tqdm)
        for result in results:
            (parameter_count, mem_allocated, max_mem_allocated, latencies,
             tflops, tflops_ckpt, compilation_times, compute_cost_file_name,
             forward_stage_layer_ids, submesh_shapes, logical_mesh_shapes,
             autosharding_option_dicts) = result

            heads = [
                "Type", "Model Config", "#GPUs", "#Layers (for Auto-Layer)",
                "#Microbatch", "Remat", "Reduce-scatter", "Mean Time",
                "Std Time", "#Params", "TFLOPs", "TFLOPs (ckpt)", "Peak Mem",
                "Compute Cost File", "Layer->Stage Mapping", "Submesh Shapes",
                "Logical Mesh Shapes", "Autosharding Global Configs",
                "overwrite_global_config_dict", "compilation times"
            ]
            values = [
                model_type + "-" + pipeline_stage_mode, model_config, num_gpus,
                pipeline_mp_size, num_micro_batches, use_remat,
                prefer_reduce_scatter, f"{np.mean(latencies):.3f}s",
                f"{np.std(latencies):.3f}", f"{parameter_count/1e9:.3f}B",
                f"{tflops:.2f}", f"{tflops_ckpt:.2f}",
                f"{max_mem_allocated/GB:.3f}G", compute_cost_file_name,
                forward_stage_layer_ids, submesh_shapes, logical_mesh_shapes,
                autosharding_option_dicts, overwrite_global_config_dict,
                to_str_round(compilation_times, 2)
            ]
            write_tsv(heads, values, output_name)

        time.sleep(0.1)  # for ctrl+c to work

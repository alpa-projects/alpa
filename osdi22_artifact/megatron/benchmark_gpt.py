import argparse

from benchmark.alpa.suite_paper_manual_gpt import gpt_specs
from benchmark.util import run_cmd

# Remat, RS,   pipeline_stage_mode,   overwrite_global_config_dict
fixed_params = (True,  True, "uniform_layer_gpipe", None)

gpt_megatron_best_suite = {
1: [
    (128,  *gpt_specs["350M"],  1,   1,   1,   1,   1,  8,    False,  *fixed_params),
],
4: [
    (1024,  *gpt_specs["1.3B"],  2,  1,   1,   1,   2,  128,    False,  *fixed_params),
],
8: [
    (1024,  *gpt_specs["2.6B"],  1,   1,   1,   1,   8,  128,    False,  *fixed_params),
],
16: [
    (1024,  *gpt_specs["6.7B"],  1,   2,   1,   2,   8,  256,    False,  *fixed_params),
],
32: [
    (1024,  *gpt_specs["15B"],  1,   4,   1,   4,   8,  256,    False,  *fixed_params),
]
}

def benchmark_all(args):
    num_gpus = args.nproc_per_node * args.nnodes

    try:
        _ = gpt_megatron_best_suite[num_gpus]
    except KeyError:
        print(f"No available benchmark suite for {args.suite} with {num_gpus} GPUs.")
        exit()
    output_name = f"megatron_results_e2e"
    model = "gpt"

    for case in gpt_megatron_best_suite[num_gpus]:
        case_str = str((model,) + case)

        if args.nnodes == 1:
            # Single node
            ret = run_cmd('python3 -m torch.distributed.launch '
                         f'--nproc_per_node {args.nproc_per_node} '
                         'benchmark_gpt_one_case.py '
                          f'"{case_str}" '
                          f'{output_name}')
        else:
            # Multiple nodes
            ret = run_cmd('python3 -m torch.distributed.launch '
                         f'--nproc_per_node {args.nproc_per_node} '
                         f'--nnodes {args.nnodes} '
                         f'--node_rank {args.node_rank} '
                         f'--master_addr {args.master_addr} '
                         f'--master_port {args.master_port} '
                         'benchmark_gpt_one_case.py '
                         f'"{case_str}" '
                         f'{output_name}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    args = parser.parse_args()
    benchmark_all(args)

import argparse
from datetime import datetime

from util import run_cmd

from benchmark.alpa import suite_manual_gpt

benchmark_suites = {
    "gpt.tmp": suite_manual_gpt.tmp_suite,
    #"gpt.grid_search_manual": suite_manual_gpt.grid_search_manual,
}

def benchmark_all(args):
    num_gpus = args.nproc_per_node * args.nnodes

    try:
        _ = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        print(f"No available benchmark suite for {args.suite} with {num_gpus} GPUs.")
        exit()
    output_name = args.exp_name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model = args.suite.split(".")[0]

    for case in benchmark_suites[args.suite][num_gpus]:
        case = tuple(tuple(x) if isinstance(x, tuple) else x for x in case)
        case_str = str((model,) + case)

        if args.nnodes == 1:
            # Single node
            ret = run_cmd('python3 -m torch.distributed.launch '
                         f'--nproc_per_node {args.nproc_per_node} '
                         'benchmark_gpt_bert_one_case.py '
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
                         'benchmark_gpt_bert_one_case.py '
                         f'"{case_str}" '
                         f'{output_name}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    parser.add_argument("--suite", type=str, default="gpt.tmp")
    parser.add_argument("--exp_name", type=str, default="")
    args = parser.parse_args()

    benchmark_all(args)

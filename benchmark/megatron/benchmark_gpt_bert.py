import argparse
from datetime import datetime

from util import run_cmd

from benchmark.alpa.paper_manual_gpt_suite import paper_gpt_suite, test_gpt_suite, gpt_specs

_ = None

# B = global_batch_size, S = seq_len,
# H = hidden_size, L = num_layers, V = vocab_size, #head = num_heads,
# DP = data_parallel, TP = tensor_model_parallel, PP = pipeline_model_parallel,
# NB = num_micro_batches
# DI = ddp_implementation, CK = checkpoint_activations

default_benchmark_suite = {
1 : [
    # B,       model,         LD0, LD1, PD0, PD1, PP, NB, FD,  Remat, Auto-layer, Auto-stage
    (32,  *gpt_specs["125M"], 1,   1,   _,   _,   1,  1,  _,   True,  _, _),
    (40,  *gpt_specs["125M"], 1,   1,   _,   _,   1,  1,  _,   True,  _, _),

    (24,  *gpt_specs["350M"], 1,   1,   _,   _,   1,  1,  _,   True,  _, _),
    (32,  *gpt_specs["350M"], 1,   1,   _,   _,   1,  1,  _,   True,  _, _),

    # with acc
    (512, *gpt_specs["125M"], 1,   1,   _,   _,   1,  16, _,   True,  _, _),
    (504, *gpt_specs["350M"], 1,   1,   _,   _,   1,  21, _,   True,  _, _),
],

2 : [
    # B,       model,         LD0, LD1, PD0, PD1, PP, NB, FD,  Remat, Auto-layer, Auto-stage
    (32,  *gpt_specs["350M"], 2,   1,   _,   _,   1,  1,  _,   True, _, _),
    (40,  *gpt_specs["350M"], 2,   1,   _,   _,   1,  1,  _,   True, _, _),

    (24,  *gpt_specs["350M"], 1,   2,   _,   _,   1,  1,  _,   True, _, _),
    (32,  *gpt_specs["350M"], 1,   2,   _,   _,   1,  1,  _,   True, _, _),

    (2,   *gpt_specs["760M"], 2,   1,   _,   _,   1,  1,  _,   True, _, _),
    (16,  *gpt_specs["760M"], 1,   2,   _,   _,   1,  1,  _,   True, _, _),
    (24,  *gpt_specs["760M"], 1,   2,   _,   _,   1,  1,  _,   True, _, _),

    # with acc
    (512, *gpt_specs["350M"], 2,   1,   _,   _,   1,  16, _,   True, _, _),
    (504, *gpt_specs["350M"], 1,   2,   _,   _,   1,  21, _,   True, _, _),
    (512, *gpt_specs["350M"], 1,   1,   _,   _,   2,  32, _,   True, _, _),

    (512, *gpt_specs["760M"], 1,   2,   _,   _,   1,  32, _,   True, _, _),
    (512, *gpt_specs["760M"], 1,   1,   _,   _,   2,  32, _,   True, _, _),
],

4: [
    # B,       model,         LD0, LD1, PD0, PD1, PP, NB, FD,  Remat, Auto-layer, Auto-stage
    (40,  *gpt_specs["760M"], 2,   2,   _,   _,   1,  1,  _,   True, _, _),
    (48,  *gpt_specs["760M"], 2,   2,   _,   _,   1,  1,  _,   True, _, _),
    (32,  *gpt_specs["760M"], 1,   4,   _,   _,   1,  1,  _,   True, _, _),
    (40,  *gpt_specs["760M"], 1,   4,   _,   _,   1,  1,  _,   True, _, _),

    (8,   *gpt_specs["1.3B"], 2,   2,   _,   _,   1,  1,  _,   True, _, _),
    (16,  *gpt_specs["1.3B"], 2,   2,   _,   _,   1,  1,  _,   True, _, _),
    (16,  *gpt_specs["1.3B"], 1,   4,   _,   _,   1,  1,  _,   True, _, _),
    (32,  *gpt_specs["1.3B"], 1,   4,   _,   _,   1,  1,  _,   True, _, _),

    # with acc
    (520, *gpt_specs["760M"], 2,   2,   _,   _,   1,  13, _,   True, _, _),
    (512, *gpt_specs["760M"], 1,   4,   _,   _,   1,  16, _,   True, _, _),
    (504, *gpt_specs["760M"], 1,   1,   _,   _,   4,  21, _,   True, _, _),

    (512, *gpt_specs["1.3B"], 2,   2,   _,   _,   1,  64, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 1,   4,   _,   _,   1,  32, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 1,   1,   _,   _,   4,  64, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 2,   1,   _,   _,   2,  64, _,   True, _, _),
],

8: [
    # B,       model,         LD0, LD1, PD0, PD1, PP, NB, FD,  Remat, Auto-layer, Auto-stage
    (24,  *gpt_specs["1.3B"], 4,   2,   _,   _,   1,  1,  _,   True, _, _),
    (32,  *gpt_specs["1.3B"], 4,   2,   _,   _,   1,  1,  _,   True, _, _),
    (40,  *gpt_specs["1.3B"], 2,   4,   _,   _,   1,  1,  _,   True, _, _),
    (48,  *gpt_specs["1.3B"], 2,   4,   _,   _,   1,  1,  _,   True, _, _),
    (32,  *gpt_specs["1.3B"], 1,   8,   _,   _,   1,  1,  _,   True, _, _),
    (40,  *gpt_specs["1.3B"], 1,   8,   _,   _,   1,  1,  _,   True, _, _),

    (4,   *gpt_specs["2.6B"], 4,   2,   _,   _,   1,  1,  _,   True, _, _),
    (8,   *gpt_specs["2.6B"], 2,   4,   _,   _,   1,  1,  _,   True, _, _),
    (16,  *gpt_specs["2.6B"], 2,   4,   _,   _,   1,  1,  _,   True, _, _),
    (16,  *gpt_specs["2.6B"], 1,   8,   _,   _,   1,  1,  _,   True, _, _),
    (24,  *gpt_specs["2.6B"], 1,   8,   _,   _,   1,  1,  _,   True, _, _),

    # with acc
    (504, *gpt_specs["1.3B"], 4,   2,   _,   _,   1,  21, _,   True, _, _),
    (520, *gpt_specs["1.3B"], 2,   4,   _,   _,   1,  13, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 1,   8,   _,   _,   1,  16, _,   True, _, _),

    (512, *gpt_specs["1.3B"], 1,   1,   _,   _,   8,  64, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 1,   1,   _,   _,   8,  32, _,   True, _, _),

    (512, *gpt_specs["1.3B"], 2,   1,   _,   _,   4,  64, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 2,   1,   _,   _,   4,  32, _,   True, _, _),
    (504, *gpt_specs["1.3B"], 2,   1,   _,   _,   4,  21, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 2,   1,   _,   _,   4,  16, _,   True, _, _),

    (512, *gpt_specs["1.3B"], 1,   2,   _,   _,   4,  64, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 1,   2,   _,   _,   4,  32, _,   True, _, _),
    (504, *gpt_specs["1.3B"], 1,   2,   _,   _,   4,  21, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 1,   2,   _,   _,   4,  16, _,   True, _, _),

    (512, *gpt_specs["1.3B"], 4,   1,   _,   _,   2,  64, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 4,   1,   _,   _,   2,  32, _,   True, _, _),
    (504, *gpt_specs["1.3B"], 4,   1,   _,   _,   2,  21, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 4,   1,   _,   _,   2,  16, _,   True, _, _),

    (512, *gpt_specs["1.3B"], 1,   4,   _,   _,   2,  64, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 1,   4,   _,   _,   2,  32, _,   True, _, _),
    (504, *gpt_specs["1.3B"], 1,   4,   _,   _,   2,  21, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 1,   4,   _,   _,   2,  16, _,   True, _, _),

    (512, *gpt_specs["1.3B"], 2,   2,   _,   _,   2,  64, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 2,   2,   _,   _,   2,  32, _,   True, _, _),
    (504, *gpt_specs["1.3B"], 2,   2,   _,   _,   2,  21, _,   True, _, _),
    (512, *gpt_specs["1.3B"], 2,   2,   _,   _,   2,  16, _,   True, _, _),
]
}

benchmark_suites = {
    "default": default_benchmark_suite,
    "paper_gpt": paper_gpt_suite,
    "test_gpt": test_gpt_suite,
}

def benchmark_all(args):
    num_gpus = args.nproc_per_node * args.nnodes

    try:
        _ = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        print(f"No available benchmark suite for {args.suite} with {num_gpus} GPUs.")
        exit()
    output_name = args.exp_name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    for case in benchmark_suites[args.suite][num_gpus]:
        case_str = str((args.model,) + case)

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
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    parser.add_argument("--suite", type=str, default="paper_gpt")
    parser.add_argument("--exp_name", type=str, default="")
    args = parser.parse_args()

    benchmark_all(args)

import argparse
from datetime import datetime

from util import run_cmd

from benchmark.parax.paper_manual_gpt_suite import paper_gpt_suite, test_gpt_suite

# B = global_batch_size, S = seq_len,
# H = hidden_size, L = num_layers, V = vocab_size, #head = num_heads,
# DP = data_parallel, TP = tensor_model_parallel, PP = pipeline_model_parallel,
# NB = num_micro_batches
# DI = ddp_implementation, CK = checkpoint_activations

default_benchmark_suite = {

1: [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, DI, CK
    (16,  512,  1024, 24, 1024//64,  32000, 1,  1,  1,  1,  1,  0),
    (8,   1024, 1536, 16, 1536//96,  32000, 1,  1,  1,  1,  1,  0),
],

4: [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, DI, CK
    # (8,   1024, 1536, 16, 1536//96,  32000, 1,  1,  1,  1,  1,  0),
    # (16,  512,  1024, 24, 1024//64,  32000, 1,  1,  1,  1,  1,  0),
    # (32,  512,  1024, 24, 1024//64,  32000, 4,  1,  1,  1,  1,  0),
    # (32,  512,  1024, 24, 1024//64,  32000, 1,  4,  1,  1,  1,  0),
    (8,  1024,  1024, 24, 1024//64,  51200, 1,  1,  4,  1,  True, False),
    # # smaller
    # (32,  512,  1024, 24, 1024//64,  32000, 2,  2,  1,  1,  1,  0),
    # (32,  512,  1024, 24, 1024//64,  32000, 2,  1,  2,  1,  1,  0),
    # (32,  512,  1024, 24, 1024//64,  32000, 1,  2,  2,  1,  1,  0),
    #
    # # V=32000, others are GPT-2
    # (32,  1024,  1024, 24, 1024//64,  32000, 1,  1,  4,  1,  1,  0),
    #
    # (32,  1024,  1024, 24, 1024//64,  32000, 1,  1,  4,  1,  1,  0),
    #
    # # (32,   1024, 1536, 16, 1536//96,  32000, 1,  1,  1,  1,  1,  0),
    #
    # (32,  1024,  1024, 24, 1024//64,  51200, 1,  1,  4,  1,  1,  0),

],

8: [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, DI, CK
    # (128, 512,  1024, 24, 1024//64,  32000, 8,  1,  1,  1,  1,  0),
    # (256, 512,  1024, 24, 1024//64,  32000, 8,  1,  1,  2,  1,  0),
    # (8,   1024, 4096, 20, 4096//128, 32000, 1,  8,  1,  1,  1,  0),
    # (16,  1024, 4096, 20, 4096//128, 32000, 1,  8,  1,  2,  1,  0),
    # (256, 1024, 4096, 20, 4096//128, 32000, 1,  8,  1,  32, 1,  0),
    # (8,  1024,  1024, 24, 1024//64,  51200, 1,  1,  8,  1,  True, False),
    # (32,  1024,  1024, 24, 1024//64, 51200, 4,   1,   2,   8,  True, True),
    # (2048,  1024,  1024, 24, 1024//64, 51200, 4,   1,   2,   128,  True, True), # 30 TFLOPs
    # (2048,  1024,  1024, 24, 1024//64, 51200, 1,   4,   2,   128,  True, True), # 21 TFLOPs
    # 0.587
    # (32,  1024,  1024, 24, 1024//64, 51200, 4,   1,   1,   4,   2,  8,   True, True, True, False),
    # 0.666
    # (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  8,   True, True, False, False),
    # (32,  1024,  1024, 6, 1024//64, 51200, 4,   1,   1,   4,   2,  8,   True, True, False, False), # 0.198
    # (32,  1024,  1024, 6, 1024//64, 51200, 1,   4,   1,   4,   2,  8,   True, True, False, False), # 0.221
    # (32,  1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  8,   True, True, False, False), # 0.323
    # (32,  1024,  1024, 12, 1024//64, 51200, 1,   4,   1,   4,   2,  8,   True, True, False, False), # 0.380
    (64,  1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  16,   True, True, False, False), # 0.323
    (64,  1024,  1024, 12, 1024//64, 51200, 1,   4,   1,   4,   2,  16,   True, True, False, False), # 0.380
],

16: [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, DI, CK
    (256, 512,  1024, 24, 1024//64,  32000, 16, 1,  1,  1,  1,  0),
    (512, 512,  1024, 24, 1024//64,  32000, 16, 1,  1,  2,  1,  0),
    (16,  1024, 4096, 20, 4096//128, 32000, 2,  8,  1,  1,  1,  0),
    (256, 1024, 4096, 20, 4096//128, 32000, 2,  8,  1,  16, 1,  0),
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

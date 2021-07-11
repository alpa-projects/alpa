import argparse
import os


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


# B = Batch size, S = seq_len, H = hidden size, L = num_layers,
# #head = num_heads, DP = dp_size, TMP = tensor_mp_size, DDP = ddp_implementation

benchmark_suite_single_node = [
    # B, S,    H,    L, #head,    V,     DP, TMP, DDP
    (16, 1024, 1536, 6, 1536//96, 51200, 4,  1,   0),
    (16, 1024, 1536, 6, 1536//96, 51200, 2,  2,   0),
    (16, 1024, 1536, 6, 1536//96, 51200, 1,  4,   0),

    (8,  512,  3072, 6, 3072//96, 51200, 4,  1,   0),
    (8,  512,  3072, 6, 3072//96, 51200, 2,  2,   0),
    (8,  512,  3072, 6, 3072//96, 51200, 1,  4,   0),
]

benchmark_suite_multi_node = [
    # B, S,    H,    L, #head,     V,     DP, TMP, DDP
    (32, 1024, 1536, 6, 1536//96,  51200, 8,  1,   0),
    (32, 1024, 1536, 6, 1536//96,  51200, 4,  2,   0),
    (32, 1024, 1536, 6, 1536//96,  51200, 2,  4,   0),
    (32, 1024, 1536, 6, 1536//96,  51200, 1,  8,   0),

    (16,  512, 3072, 6, 3072//96, 51200, 8,  1,   0),
    (16,  512, 3072, 6, 3072//96, 51200, 4,  2,   0),
    (16,  512, 3072, 6, 3072//96, 51200, 2,  4,   0),
    (16,  512, 3072, 6, 3072//96, 51200, 1,  8,   0),
]

def benchmark_all(args):
    if args.master_addr is None:
        benchmark_suite = benchmark_suite_single_node
    else:
        benchmark_suite = benchmark_suite_multi_node

    for case in benchmark_suite:
        case_str = str((args.model,) + case)

        if args.master_addr is None:
            # Single node
            ret = run_cmd('python3 -m torch.distributed.launch '
                         f'--nproc_per_node {args.nproc_per_node} '
                         'benchmark_gpt_bert_one_case.py '
                         f'"{case_str}"')
        else:
            # Multiple nodes
            ret = run_cmd('python3 -m torch.distributed.launch '
                         f'--nproc_per_node {args.nproc_per_node} '
                         f'--nnodes {args.nnodes} '
                         f'--node_rank {args.node_rank} '
                         f'--master_addr {args.master_addr} '
                         f'--master_port {args.master_port} '
                         'benchmark_gpt_bert_one_case.py '
                         f'"{case_str}"')

        #if ret != 0:
        #    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--nnodes", type=str)
    parser.add_argument("--node_rank", type=str)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    args = parser.parse_args()

    benchmark_all(args)


import argparse
import os


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


# B = Batch size, S = seq_len, H = hidden size, L = num_layers,
# #head = num_heads, DP = dp_size, TMP = tensor_mp_size, DP_IMP = ddp_implementation

benchmark_suite_1_gpu = [
    # B, S,   H,    L,  #head,    V,     DP, TMP, DP_IMP
    (4, 512,  1024, 22, 1024//64, 51200, 1,  1,   1),  # bert-large
    (4, 1024, 1536, 6,  1536//96, 51200, 1,  1,   1),  # megatron 1.2B
]

benchmark_suite_4_gpu = [
    # B, S,    H,    L,  #head,    V,     DP, TMP, DP_IMP
    (16, 512,  1024, 22, 1024//64, 51200, 4,  1,   1),
    (16, 512,  1024, 22, 1024//64, 51200, 2,  2,   1),
    (16, 512,  1024, 22, 1024//64, 51200, 1,  4,   1),

    (4,  1024, 3072, 8,  3072//96, 51200, 4,  1,   1),
    (4,  1024, 3072, 8,  3072//96, 51200, 2,  2,   1),
    (4,  1024, 3072, 8,  3072//96, 51200, 1,  4,   1),
]

benchmark_suite_8_gpu = [
    # B, S,    H,    L,  #head,    V,     DP, TMP, DP_IMP
    (32, 512,  1024, 22, 1024//64, 51200, 8,  1,   1),
    (32, 512,  1024, 22, 1024//64, 51200, 4,  2,   1),
    (32, 512,  1024, 22, 1024//64, 51200, 2,  4,   1),
    (32, 512,  1024, 22, 1024//64, 51200, 1,  8,   1),

    (8,  1024, 3072, 8,  3072//96, 51200, 8,  1,   1),
    (8,  1024, 3072, 8,  3072//96, 51200, 4,  2,   1),
    (8,  1024, 3072, 8,  3072//96, 51200, 2,  4,   1),
    (8,  1024, 3072, 8,  3072//96, 51200, 1,  8,   1),
]

def benchmark_all(args):
    if args.nproc_per_node == 1:
        benchmark_suite = benchmark_suite_1_gpu
    elif args.nnodes is None or args.nnodes == 1:
        benchmark_suite = benchmark_suite_4_gpu
    else:
        benchmark_suite = benchmark_suite_8_gpu

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


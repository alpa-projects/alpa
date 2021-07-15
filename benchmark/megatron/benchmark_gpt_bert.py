import argparse
import os


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


# B = Batch size, S = seq_len, H = hidden size, L = num_layers,
# #head = num_heads, DP = dp_size, TMP = tensor_mp_size, DP_IMP = ddp_implementation

benchmark_suite_1_gpu = [
    # B, S,   H,    L,  #head,    V,     DP, TMP, DP_IMP
    (32, 512,  1024, 24, 1024//64, 51200, 1,  1,   1),  # bert-large

    #(8, 1024, 1536, 40, 1536//96, 51200, 1,  1,   1),  # megatron 1.2B
]

benchmark_suite_4_gpu = [
    # B, S,    H,    L,  #head,    V,     DP, TMP, DP_IMP
]

benchmark_suite_8_gpu = [
    # B, S,    H,    L,  #head,    V,     DP, TMP, DP_IMP
    (256, 512,  1024, 24, 1024//64, 51200, 8,  1,   1),

    #(8,  1024, 3072, 72,  3072//96, 51200, 1,  8,   1),
]

benchmark_suite_16_gpu = [
    # B, S,    H,    L,  #head,    V,     DP, TMP, DP_IMP
    #(512, 512,  1024, 24, 1024//64, 51200, 16,  1,   1),

    (16,  1024, 3072, 8,  3072//96, 51200, 2,  8,   1),

    #(16,  1024, 3072, 72,  3072//96, 51200, 2,  8,   1),
    #(32,  1024, 3072, 72,  3072//96, 51200, 2,  8,   1),
    #(64,  1024, 3072, 72,  3072//96, 51200, 2,  8,   1),
]


def benchmark_all(args):
    num_gpus = args.nproc_per_node * args.nnodes

    benchmark_suites = {
        1 : benchmark_suite_1_gpu,
        4 : benchmark_suite_4_gpu,
        8 : benchmark_suite_8_gpu,
        16 : benchmark_suite_16_gpu,
    }

    for case in benchmark_suites[num_gpus]:
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
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    args = parser.parse_args()

    benchmark_all(args)


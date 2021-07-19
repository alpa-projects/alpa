import argparse

from util import run_cmd

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers,
# #head = num_heads, DP = dp_size, TMP = tensor_mp_size, DPI = ddp_implementation,

benchmark_suite_4_gpu = [
    # B,  S,    H,    L,  #head,     DP, TMP, DPI
    (32,  1024, 1536, 3,  1536//96,  4,  1,   1,),
    (32,  1024, 1536, 3,  1536//96,  2,  2,   1,),
    (32,  1024, 1536, 3,  1536//96,  1,  4,   1,),

    (32,  128,  5120, 2,  5120//128, 4,  1,   1,),
    (32,  128,  5120, 2,  5120//128, 2,  2,   1,),
    (32,  128,  5120, 2,  5120//128, 1,  4,   1,),
]

benchmark_suite_8_gpu = [
    # B,  S,    H,    L,  #head,     DP, TMP, DPI
    #(32,  1024, 1536, 4,  1536//96,  8,  1,   1,),
    #(32,  1024, 1536, 4,  1536//96,  4,  2,   1,),
    #(32,  1024, 1536, 4,  1536//96,  2,  4,   1,),
    #(32,  1024, 1536, 4,  1536//96,  1,  8,   1,),

    #(32,  128,  5120, 3,  5120//128, 8,  1,   1,),
    #(32,  128,  5120, 3,  5120//128, 4,  2,   1,),
    #(32,  128,  5120, 3,  5120//128, 2,  4,   1,),
    (32,  128,  5120, 3,  5120//128, 1,  8,   1,),
]


def benchmark_all(args):
    num_gpus = args.nproc_per_node * args.nnodes

    benchmark_suites = {
        4 : benchmark_suite_4_gpu,
        8 : benchmark_suite_8_gpu,
    }

    for case in benchmark_suites[num_gpus]:
        case_str = str(case)

        if args.master_addr is None:
            # Single node
            ret = run_cmd('python3 -m torch.distributed.launch '
                         f'--nproc_per_node {args.nproc_per_node} '
                         'benchmark_transformer_layer_one_case.py '
                         f'"{case_str}"')
        else:
            # Multiple nodes
            ret = run_cmd('python3 -m torch.distributed.launch '
                         f'--nproc_per_node {args.nproc_per_node} '
                         f'--nnodes {args.nnodes} '
                         f'--node_rank {args.node_rank} '
                         f'--master_addr {args.master_addr} '
                         f'--master_port {args.master_port} '
                         'benchmark_transformer_layer_one_case.py '
                         f'"{case_str}"')

        #if ret != 0:
        #    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    args = parser.parse_args()

    benchmark_all(args)

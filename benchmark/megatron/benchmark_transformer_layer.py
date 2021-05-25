import argparse
import os


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


benchmark_suite_single_node = [
    # Batch size, seq_len, hidden size, num_layers, num_heads, dp_size, tensor_mp_size, ddp_impl
    (32,          1024,    1536,        3,          1536//96,  4,       1,              0,),
    (32,          1024,    1536,        3,          1536//96,  2,       2,              0,),
    (32,          1024,    1536,        3,          1536//96,  1,       4,              0,),

    (32,          128,     5120,        2,          5120//128, 4,       1,              0,),
    (32,          128,     5120,        2,          5120//128, 2,       2,              0,),
    (32,          128,     5120,        2,          5120//128, 1,       4,              0,),
]

benchmark_suite_multi_node = [
    # Batch size, seq_len, hidden size, num_layers, num_heads, dp_size, tensor_mp_size, ddp_impl
    (32,          128,     1536,        1,          1536//96,  8,       1,              0,),

    (32,          1024,    1536,        4,          1536//96,  8,       1,              0,),
    (32,          1024,    1536,        4,          1536//96,  4,       2,              0,),
    (32,          1024,    1536,        4,          1536//96,  2,       4,              0,),
    (32,          1024,    1536,        4,          1536//96,  1,       8,              0,),

    (32,          128,     5120,        3,          5120//128, 8,       1,              0,),
    (32,          128,     5120,        3,          5120//128, 4,       2,              0,),
    (32,          128,     5120,        3,          5120//128, 2,       4,              0,),
    (32,          128,     5120,        3,          5120//128, 1,       8,              0,),

    (32,          1024,    1536,        4,          1536//96,  8,       1,              1,),
    (32,          1024,    1536,        4,          1536//96,  4,       2,              1,),
    (32,          1024,    1536,        4,          1536//96,  2,       4,              1,),
    (32,          1024,    1536,        4,          1536//96,  1,       8,              1,),

    (32,          128,     5120,        3,          5120//128, 8,       1,              1,),
    (32,          128,     5120,        3,          5120//128, 4,       2,              1,),
    (32,          128,     5120,        3,          5120//128, 2,       4,              1,),
    (32,          128,     5120,        3,          5120//128, 1,       8,              1,),
]


def benchmark_all(args):
    if args.master_addr is None:
        benchmark_suite = benchmark_suite_single_node
    else:
        benchmark_suite = benchmark_suite_multi_node

    for case in benchmark_suite:
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
    parser.add_argument("--nnodes", type=str)
    parser.add_argument("--node_rank", type=str)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    args = parser.parse_args()

    benchmark_all(args)


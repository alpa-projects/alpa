import argparse
import os


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


benchmark_suits_single_node = [
    # Batch size, seq_len, hidden size, num_layers, num_heads, dp_size, tensor_mp_size, ddp_impl
    (16,          1024,    1536,        3,          1536//96,  4,       1,              0,),
    (16,          1024,    1536,        3,          1536//96,  2,       2,              0,),
    (16,          1024,    1536,        3,          1536//96,  1,       4,              0,),

    (8,           256,     2304,        3,          2304//96,  4,       1,              0,),
    (8,           256,     2304,        3,          2304//96,  2,       2,              0,),
    (8,           256,     2304,        3,          2304//96,  1,       4,              0,),

    (16,          1024,    1536,        3,          1536//96,  4,       1,              1,),
    (16,          1024,    1536,        3,          1536//96,  2,       2,              1,),
    (16,          1024,    1536,        3,          1536//96,  1,       4,              1,),

    (8,           256,     2304,        3,          2304//96,  4,       1,              1,),
    (8,           256,     2304,        3,          2304//96,  2,       2,              1,),
    (8,           256,     2304,        3,          2304//96,  1,       4,              1,),
]

benchmark_suits_multi_node = [
    # Batch size, seq_len, hidden size, num_layers, num_heads, dp_size, tensor_mp_size, ddp_impl
    (16,          1024,    2304,        3,          2304//96,  8,       1,              0,),
    (16,          1024,    2304,        3,          2304//96,  4,       2,              0,),
    (16,          1024,    2304,        3,          2304//96,  2,       4,              0,),
    (16,          1024,    2304,        3,          2304//96,  1,       8,              0,),

    (8,           256,     2304,        6,          2304//96,  8,       1,              0,),
    (8,           256,     2304,        6,          2304//96,  4,       2,              0,),
    (8,           256,     2304,        6,          2304//96,  2,       4,              0,),
    (8,           256,     2304,        6,          2304//96,  1,       8,              0,),
]


def benchmark_all(args):
    if args.master_address is None:
        benchmark_suits = benchmark_suits_single_node
    else:
        benchmark_suits = benchmark_suits_multi_node

    for case in benchmark_suits:
        case_str = str(case)

        if args.master_address is None:
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
                         f'--master_address {args.master_address} '
                         f'--master_port {args.master_port} '
                         'benchmark_transformer_layer_one_case.py '
                         f'"{case_str}"')

        if ret != 0:
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--nnodes", type=str)
    parser.add_argument("--node_rank", type=str)
    parser.add_argument("--master_address", type=str)
    parser.add_argument("--master_port", type=str)
    args = parser.parse_args()

    benchmark_all(args)


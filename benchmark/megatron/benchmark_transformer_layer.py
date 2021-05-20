import os


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


benchmark_suits = [
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


def benchmark_all():
    for case in benchmark_suits:
        nproc_per_node = 4
        case_str = str(case)
        ret = run_cmd('python3 -m torch.distributed.launch '
                     f'--nproc_per_node {nproc_per_node} '
                     'benchmark_transformer_layer_one_case.py '
                     f'"{case_str}"')
        if ret != 0:
            return


if __name__ == "__main__":
    benchmark_all()


import os

def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)

benchmark_suits = [
    # Batch size, seq_len, hidden size, num_layers, num_heads, dp_size, tensor_mp_size,
    (16,          1024,    2304,        4,          2304//96,  4,       1),
    (16,          1024,    2304,        4,          2304//96,  2,       2),
    (16,          1024,    2304,        4,          2304//96,  1,       4),

    # Batch size, seq_len, hidden size, num_layers, num_heads, dp_size, tensor_mp_size,
    (8,           256,     2304,        4,          2304//96,  4,       1),
    (8,           256,     2304,        4,          2304//96,  2,       2),
    (8,           256,     2304,        4,          2304//96,  1,       4),
]

def benchmark_mlp():
    for case in benchmark_suits:
        nproc_per_node = 4
        case_str = str(case)
        ret = run_cmd('python3 -m torch.distributed.launch '
                     f'--nproc_per_node {nproc_per_node} '
                     'benchmark_mlp_one_case.py '
                     f'"{case_str}"')
        if ret != 0:
            return

if __name__ == "__main__":
    benchmark_mlp()


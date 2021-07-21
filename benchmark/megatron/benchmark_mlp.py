import argparse

from util import run_cmd

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers,
# #head = num_heads, DP = dp_size, TMP = tensor_mp_size, DPI = ddp_implementation,

benchmark_suite_4_gpu = [
    # B,  S,    H,    L,  #head,     DP, TMP, DPI
    (32,  1024, 2304, 4,  2304//96,  4,  1,   1),
    (32,  1024, 2304, 4,  2304//96,  2,  2,   1),
    (32,  1024, 2304, 4,  2304//96,  1,  4,   1),

    # B,  S,    H,    L,  #head,     DP, TMP, DPI
    (8,   256,  5760, 4,  5760//96,  4,  1,   1),
    (8,   256,  5760, 4,  5760//96,  2,  2,   1),
    (8,   256,  5760, 4,  5760//96,  1,  4,   1),
]


def benchmark_all():
    for case in benchmark_suite_4_gpu:
        nproc_per_node = 4
        case_str = str(case)
        ret = run_cmd('python3 -m torch.distributed.launch '
                     f'--nproc_per_node {nproc_per_node} '
                     'benchmark_mlp_one_case.py '
                     f'"{case_str}"')
        if ret != 0:
            return

if __name__ == "__main__":
    benchmark_all()


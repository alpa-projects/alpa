import argparse

from util import run_cmd

# B = global_batch_size, S = seq_len,
# H = hidden_size, L = num_layers, V = vocab_size, #head = num_heads,
# DP = data_parallel, TP = tensor_model_parallel, PP = pipeline_model_parallel,
# NB = num_micro_batches
# DI = ddp_implementation, CK = checkpoint_activations

benchmark_suite_1_gpu = [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, DI, CK
    (16,  512,  1024, 24, 1024//64,  32000, 1,  1,  1,  1,  1,  0),
    (8,   1024, 1536, 16, 1536//96,  32000, 1,  1,  1,  1,  1,  0),
]

benchmark_suite_4_gpu = [
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

]

benchmark_suite_8_gpu = [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, DI, CK
    # (128, 512,  1024, 24, 1024//64,  32000, 8,  1,  1,  1,  1,  0),
    # (256, 512,  1024, 24, 1024//64,  32000, 8,  1,  1,  2,  1,  0),
    # (8,   1024, 4096, 20, 4096//128, 32000, 1,  8,  1,  1,  1,  0),
    # (16,  1024, 4096, 20, 4096//128, 32000, 1,  8,  1,  2,  1,  0),
    # (256, 1024, 4096, 20, 4096//128, 32000, 1,  8,  1,  32, 1,  0),
    # (8,  1024,  1024, 24, 1024//64,  51200, 1,  1,  8,  1,  True, False),
    # (32,  1024,  1024, 24, 1024//64, 51200, 4,   1,   2,   8,  True, True),
    (2048,  1024,  1024, 24, 1024//64, 51200, 4,   1,   2,   128,  True, True), # 30 TFLOPs
    (2048,  1024,  1024, 24, 1024//64, 51200, 1,   4,   2,   128,  True, True), # 21 TFLOPs

]

benchmark_suite_16_gpu = [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, DI, CK
    (256, 512,  1024, 24, 1024//64,  32000, 16, 1,  1,  1,  1,  0),
    (512, 512,  1024, 24, 1024//64,  32000, 16, 1,  1,  2,  1,  0),
    (16,  1024, 4096, 20, 4096//128, 32000, 2,  8,  1,  1,  1,  0),
    (256, 1024, 4096, 20, 4096//128, 32000, 2,  8,  1,  16, 1,  0),
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

        if args.nnodes == 1:
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


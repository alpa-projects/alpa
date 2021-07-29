import argparse
import os

from util import run_cmd

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size,
# #head = num_heads, DP = dp_size, TMP = tensor_mp_size, DPI = ddp_implementation,
# CK = checkpoint_activations, DS = use_deepspeed

benchmark_suite_1_gpu = [
    # B,  S,    H,    L,  #head,     V,     DP, TMP, DPI, CK, DS
    (16,  512,  1024, 10, 1024//64,  25600, 1,  1,   1,   0,  1),
    (8,   1024, 1536, 10, 1536//96,  25600, 1,  1,   1,   0,  1),
]

benchmark_suite_4_gpu = [
    # B,  S,    H,    L,  #head,     V,     DP, TMP, DPI, CK
]

benchmark_suite_8_gpu = [
    # B,  S,    H,    L,  #head,     V,     DP, TMP, DPI, CK, DS
    (256, 512,  1024, 10, 1024//64,  25600, 8,  1,   1,   0,  1),
    (8,   1024, 4096, 10, 4096//128, 25600, 1,  8,   1,   0,  0),
    (8,   1024, 4096, 10, 4096//128, 25600, 8,  1,   1,   0,  1),
]

benchmark_suite_16_gpu = [
    # B,  S,    H,    L,  #head,     V,     DP, TMP, DPI, CK, DS
    (512, 512,  1024, 10, 1024//64,  25600, 16, 1,   1,   0,  1),
    (16,  1024, 4096, 10, 4096//128, 25600, 2,  8,   1,   0,  1),
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
        batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,\
        dp_size, tensor_mp_size, dp_imp, checkpoint_activations, use_deepspeed = case

        assert dp_size * tensor_mp_size == num_gpus
        assert batch_size % dp_size == 0

        gpt_options = (
            f"--model-parallel-size {tensor_mp_size} "
            f"--num-layers {num_layers} "
            f"--hidden-size {hidden_size} "
            f"--num-attention-heads {num_heads} "
            f"--batch-size {batch_size // dp_size} "
            f"--seq-length {seq_len} "
            f"--max-position-embeddings {seq_len} "
            f"--train-iters 10 "
            f"--train-data webtext "
            f"--lazy-loader "
            f"--tokenizer-type GPT2BPETokenizer "
            f"--split 100,0,0 "
            f"--distributed-backend nccl "
            f"--lr 0.00015 "
            f"--no-load-optim "
            f"--lr-decay-style cosine "
            f"--weight-decay 1e-2 "
            f"--clip-grad 1.0 "
            f"--warmup .01 "
            f"--checkpoint-num-layers 1 "
            f"--fp16 "
            f"--cache-dir /tmp/cache_dir "
            f"--log-interval 1 "
        )

        if checkpoint_activations:
            gpt_options += "--checkpoint-activations --deepspeed-activation-checkpointing "

        if use_deepspeed:
            gpt_options += "--deepspeed --deepspeed_config ds_zero2_config.json "

        if args.nnodes == 1:
            # Single node
            work_dir= os.environ["DEEPSPEED_PATH"] + "/DeepSpeedExamples/Megatron-LM/"
            ret = run_cmd(f"PYTHONPATH={work_dir} VOCAB_SIZE={vocab_size} deepspeed "
                          f"--master_port {args.master_port} "
                          f"--num_nodes {args.nnodes} "
                          f"--num_gpus {args.nproc_per_node} "
                          f"pretrain_gpt2.py {gpt_options}")
        else:
            # Multiple nodes
            raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--master_port", type=int, default=6000)
    args = parser.parse_args()

    benchmark_all(args)


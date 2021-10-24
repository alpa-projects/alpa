import argparse
import os
import random

from util import run_cmd

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size,
# #head = num_heads, DP = dp_size, TMP = tensor_mp_size, NB = num_micro_batches,
# CK = checkpoint_activations, DS = use_deepspeed

benchmark_suite_1_gpu = [
    #B,    S,    H,    L,  #head,     V,     DP, TMP, NB, CK, DS
    (16,   512,  1024, 10, 1024//64,  25600, 1,  1,   1,  0,  1),
    (8,    1024, 1536, 10, 1536//96,  25600, 1,  1,   1,  0,  1),
]

benchmark_suite_8_gpu = [
    #B,    S,    H,    L,  #head,     V,     DP, TMP, NB, CK, DS
    (256,  512,  1024, 10, 1024//64,  25600, 8,  1,   1,  0,  1),
    (8,    1024, 4096, 10, 4096//128, 25600, 1,  8,   1,  0,  1),
    (8,    1024, 4096, 10, 4096//128, 25600, 8,  1,   1,  0,  1),
]

benchmark_suite_16_gpu = [
    #B,    S,    H,    L,  #head,     V,     DP, TMP, NB, CK, DS
    (512,  512,  1024, 10, 1024//64,  25600, 16, 1,   1,  0,  1),
    (2048, 512,  1024, 10, 1024//64,  25600, 16, 1,   4,  0,  1),
    (16,   1024, 4096, 10, 4096//128, 25600, 2,  8,   1,  0,  1),
    (64,   1024, 4096, 10, 4096//128, 25600, 2,  8,   4,  0,  1),
    (16,   1024, 4096, 10, 4096//128, 25600, 16, 1,   1,  0,  1),
    (64,   1024, 4096, 10, 4096//128, 25600, 16, 1,   4,  0,  1),
]


def update_ds_config(filename, gradient_accumulation_steps):
    lines = list(open(filename))

    for i in range(len(lines)):
        if "gradient_accumulation_steps" in lines[i]:
            idx = lines[i].index(":")
            lines[i] = lines[i][:idx] + f": {gradient_accumulation_steps},\n"

    with open(filename, "w") as fout:
        fout.writelines(lines)


def benchmark_all(args):
    num_gpus = args.nproc_per_node * args.nnodes

    benchmark_suites = {
        1 : benchmark_suite_1_gpu,
        8 : benchmark_suite_8_gpu,
        16 : benchmark_suite_16_gpu,
    }

    warmup_iter = 2
    bench_iter = 3
    config_file = "ds_zero_stage_2_config.json"

    for case in benchmark_suites[num_gpus]:
        batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,\
        dp_size, tensor_mp_size, num_micro_batches, checkpoint_activations, use_deepspeed\
            = case

        assert dp_size * tensor_mp_size == num_gpus
        assert batch_size % dp_size == 0
        assert batch_size & num_micro_batches == 0

        gpt_options = (
            f"--model-parallel-size {tensor_mp_size} "
            f"--num-layers {num_layers} "
            f"--hidden-size {hidden_size} "
            f"--num-attention-heads {num_heads} "
            f"--seq-length {seq_len} "
            f"--max-position-embeddings {seq_len} "
            f"--batch-size {batch_size // dp_size // num_micro_batches} "
            f"--train-iters {(warmup_iter + bench_iter) * num_micro_batches} "
            f"--lr-decay-iters 320000 "
            #f"--save $CHECKPOINT_PATH "
            #f"--load $CHECKPOINT_PATH "
            f"--data-path data/small-webtext "
            f"--vocab-file data/gpt2-vocab.json "
            f"--merge-file data/gpt2-merges.txt "
            f"--data-impl mmap "
            f"--split 949,50,1 "
            f"--distributed-backend nccl "
            f"--lr 1.5e-4 "
            f"--lr-decay-style cosine "
            f"--min-lr 1.0e-5 "
            f"--weight-decay 1e-2 "
            f"--clip-grad 1.0 "
            f"--warmup 0.01 "
            f"--log-interval 1 "
            f"--save-interval 10000 "
            f"--eval-interval 2000 "
            f"--eval-iters 0 "
            f"--fp16 "
            f"--loss-scale 1.0 "
            f"--scattered-embeddings "
            f"--split-transformers "

            # Disable fusion optimizations because this makes
            # loading too slow.
            #f"--scaled-upper-triang-masked-softmax-fusion "
            #f"--scaled-masked-softmax-fusion "
            #f"--bias-gelu-fusion "
            #f"--bias-dropout-fusion "
        )

        if use_deepspeed:
            gpt_options += (
                "--deepspeed "
                f"--deepspeed_config {config_file} "
            )
            update_ds_config(config_file, num_micro_batches)

        if checkpoint_activations:
            gpt_options += "--checkpoint-activations "
            gpt_options += "--deepspeed-activation-checkpointing "
            gpt_options += "--checkpoint-num-layers 1 "

            # Disable other checkpoint optimizations
            # gpt_options += "--partition-activations "
            # gpt_options += "--checkpoint-in-cpu "
            # gpt_options += "--synchronize-each-layer "
            # gpt_options += "--ontigious-checkpointing "

        if args.nnodes > 1:
            host_options = "--hostfile hostfile "
        else:
            host_options = ""

        work_dir= os.environ["DEEPSPEED_PATH"] + "/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/"
        ret = run_cmd(f"PYTHONPATH={work_dir} PYTHON_VOCAB_SIZE={vocab_size} deepspeed "
                      f"{host_options}"
                      f"--num_nodes {args.nnodes} "
                      f"--master_port {random.randint(10000, 20000)} "
                      f"--num_gpus {args.nproc_per_node} "
                      f"pretrain_gpt2.py {gpt_options}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc_per_node", type=int, required=True)
    args = parser.parse_args()

    benchmark_all(args)

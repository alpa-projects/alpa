import time

from datetime import datetime

import argparse
import os
import random

from util import run_cmd
from benchmark.alpa import suite_manual_moe

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# #head = num_heads, S_ = expert_group_size, E = expert_number,
# D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FD = force_data_parallel,
# CK = use_checkpoint,
# DS = use_deepspeed


benchmark_suites = {
    "paper_moe": suite_manual_moe.grid_search_manual,
    "test_moe": suite_manual_moe.tmp_suite,
}


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

    try:
        _ = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        print(f"No available benchmark suite for {args.suite} with {num_gpus} GPUs.")
        exit()
    output_name = args.exp_name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    warmup_iter = 2
    bench_iter = 3

    # MOE does not support stage 3
    config_file = "ds_zero_stage_2_moe_config.json"

    for case in benchmark_suites[args.suite][num_gpus]:
        print(">>>>>> Alpa benchmark: Working on case {}...".format(str(case)), flush=True)

        (batch_size, model_config, num_micro_batches, parallel_mode,
         parallel_args) = case
        (seq_len, hidden_size, num_layers, num_heads, vocab_size, num_expert,
         expert_group_size) = model_config

        (prefer_reduce_scatter, checkpoint_activations, dp_size, tensor_mp_size, pipeline_mp_size,
         _) = parallel_args

        # TODO (hao, zhuohan): Figure out how to set ep_size

        use_deepspeed = True

        assert dp_size * tensor_mp_size == num_gpus
        assert batch_size % dp_size == 0
        assert batch_size % num_micro_batches == 0

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

        if num_expert > 1:
            gpt_options += "--moe "
            gpt_options += "--ep-world-size {} ".format(ep_size)
            gpt_options += "--num-experts {} ".format(str(num_expert))
            gpt_options += "--top-k 2 "
            gpt_options += "--min-capacity 4 "
            gpt_options += "--noisy-gate-policy None "
            gpt_options += "--moe-param-group "
            gpt_options += "--output_name {}".format(output_name)

        if args.nnodes > 1:
            host_options = "--hostfile hostfile_{}node ".format(args.nnodes)
        else:
            host_options = ""

        work_dir= os.environ["DEEPSPEED_PATH"] + "/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/"
        ret = run_cmd(f"PYTHONPATH={work_dir} PYTHON_VOCAB_SIZE={vocab_size} deepspeed "
                      f"{host_options}"
                      f"--num_nodes {args.nnodes} "
                      f"--master_port {random.randint(30000, 40000)} "
                      f"--num_gpus {args.nproc_per_node} "
                      f"pretrain_gpt2_moe.py {gpt_options}")
        print(">>>>>> Alpa benchmark: sleep for 30 seconds before starting the next case.", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--suite", type=str, default="paper_gpt")
    parser.add_argument("--exp_name", type=str, default="none")
    args = parser.parse_args()

    benchmark_all(args)

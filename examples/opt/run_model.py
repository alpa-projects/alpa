#!/usr/bin/env python3
#!/usr/bin/env python3
import os

import argparse
import glob
import logging
import os
import sys

import torch

from metaseq import options, tasks, checkpoint_utils, utils
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as dist_utils
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap
from metaseq.distributed.stitch_fsdp_ckpt import glue_megatron_parts


from transformers import AutoTokenizer, GPT2Tokenizer
from megatron.initialize import initialize_megatron
from metaseq import checkpoint_utils
import torch

path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/1.3B"

# # just need to initialize args with something,
# # => doesn't need to correspond to the "correct" architecture for this checkpoint
# initialize_megatron(args_defaults={
#     "micro_batch_size": 1,
#     "num_layers": 12,
#     "hidden_size": 768,
#     "num_attention_heads": 12,
#     "max_position_embeddings": 2048,
#     "encoder_seq_length": 2048
# })
def worker_main(cfg: MetaseqConfig):
    vocab_file = os.path.join(path, "gpt2-vocab.json")
    merges_file = os.path.join(path, "gpt2-merges.txt")

    tokenizer = GPT2Tokenizer(vocab_file, merges_file)
    tokenizer.save_pretrained(path)

    checkpoint = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(path, "restored.pt")],
        arg_overrides={
            "vocab_filename": vocab_file,
            "merges_filename": merges_file,
        }
    )

    model = checkpoint[0][0].eval()
    model = model.cuda().half()


    # forward passes
    def single_batch_forward_logits(prompts):
        input_ids = tokenizer(prompts, return_tensors="pt").input_ids
        input_ids = torch.cat([torch.tensor([[0]]), input_ids], dim=-1)
        input_ids = input_ids.cuda()
        with torch.no_grad():
            logits = model(input_ids)[0]
        return logits

    prompts = [
        "Today is a beautiful day and I want to",
        "In the city of",
        "Paris is the capital of France and",
        "Computers and mobile phones have taken",
    ]

    print("Next word generation")
    for prompt in prompts:
        print("-------------")
        print(f"Prompt: {prompt}...\n")
        logits = single_batch_forward_logits(prompt)
        pred_next_token = torch.argmax(logits[0, -1], -1)
        next_token = tokenizer.convert_ids_to_tokens([pred_next_token])
        next_token = next_token[0].replace("Ġ", "")
        print(f"Next word: {next_token}")
        print("-------------")

def main():
    real_parser = argparse.ArgumentParser()
    real_parser.add_argument("location")
    args = real_parser.parse_args()
    # files = glob.glob(f"{args.location}/reshard*.pt")
    files = glob.glob(f"{args.location}/restore*.pt")

    MP = len(files)
    BPE_MERGES = args.location + "/gpt2-merges.txt"
    BPE_VOCAB = args.location + "/gpt2-vocab.json"


    # Skeleton out all the annoying command line args we can infer
    ARGS = [
        "--model-parallel-size",
        str(MP),
        "--distributed-world-size",
        str(MP),
        "--task",
        "language_modeling",
        "--bpe-merges",
        BPE_MERGES,
        "--bpe-vocab",
        BPE_VOCAB,
        "--bpe",
        "hf_byte_bpe",
        "--path",
        args.location + "/reshard.pt",
        "--checkpoint-shard-count",
        "1",
        "--use-sharded-state",
        args.location,
    ]
    print(ARGS)

    # build up the config file
    parser = options.get_generation_parser()
    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion=None)
    args = options.parse_args_and_arch(parser, input_args=ARGS)
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = MP
    cfg.task.vocab_filename = cfg.bpe.bpe_vocab
    cfg.task.merges_filename = cfg.bpe.bpe_merges
    # cfg.task.bpe_merges = None
    dist_utils.call_main(cfg, worker_main)

if __name__ == "__main__":
    main()
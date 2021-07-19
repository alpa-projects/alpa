import argparse
import gc
from functools import partial
import os
import sys
import timeit

import numpy as np
from megatron.utils import average_losses_across_data_parallel_group
from megatron.model import BertModel, GPTModel
from megatron import mpu, initialize_megatron, get_args
from megatron.training import train_step, setup_model_and_optimizer
import torch

from util import write_tsv, benchmark_func

GB = 1024 ** 3


def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    rank = torch.distributed.get_rank()
    device = rank % torch.cuda.device_count()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    if print_info:
        print("allocated: %.2f GB" % (allocated / GB), flush=True)
        print("reserved:  %.2f GB" % (reserved / GB), flush=True)
    return allocated


def compute_tflops(batch_size, seq_len, num_layers, hidden_size, vocab_size,
                   num_gpus, latency, checkpoint_activations=False):
    factor = 96 if checkpoint_activations else 72
    total_flop = factor * batch_size * seq_len * (hidden_size ** 2) * num_layers * \
          (1 + seq_len / (6 * hidden_size)) \
          + 6 * batch_size * seq_len * hidden_size * vocab_size
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops


def compute_parameter_count(num_layers, hidden_size, vocab_size):
    return num_layers * (
            # self-attention
            hidden_size * (3 * hidden_size + 1) + 
            hidden_size * (hidden_size + 1) + 
            # mlp
            hidden_size * (4 * hidden_size + 1) +
            hidden_size * 4 * (hidden_size + 1) +
            # layer norm
            hidden_size * 4
           ) + vocab_size * (hidden_size + 1)


def get_gpt_functions():
    args = get_args()
    micro_batch_size = args.micro_batch_size
    seq_len = args.encoder_seq_length

    def model_provider(pre_process=True, post_process=True):
        model = GPTModel(
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )
        return model

    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # Reduce loss for logging.
        #averaged_loss = average_losses_across_data_parallel_group([loss])
        averaged_loss = [0]
        return loss, {'lm loss': averaged_loss[0]}

    tokens = torch.ones((micro_batch_size, seq_len)).cuda().long()
    labels = torch.ones((micro_batch_size, seq_len)).cuda().long()
    loss_mask = torch.ones((micro_batch_size, seq_len)).cuda().int()
    attention_mask = \
        torch.ones(micro_batch_size, 1, seq_len, seq_len).cuda().bool()
    position_ids = torch.ones((micro_batch_size, seq_len)).cuda().long()

    def forward_step(data_iterator, model):
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)
        return output_tensor, partial(loss_func, loss_mask)

    return model_provider, loss_func, forward_step


def get_bert_functions():
    args = get_args()
    micro_batch_size = args.micro_batch_size
    seq_len = args.encoder_seq_length

    def model_provider(pre_process=True, post_process=True):
        num_tokentypes = 2 if args.bert_binary_head else 0
        model = BertModel(
            num_tokentypes=num_tokentypes,
            add_binary_head=args.bert_binary_head,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process)

        return model

    def loss_func(loss_mask, sentence_order, output_tensor):
        lm_loss_, sop_logits = output_tensor

        lm_loss_ = lm_loss_.float()
        loss_mask = loss_mask.float()
        lm_loss = torch.sum(
            lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

        if sop_logits is not None:
            sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                                       sentence_order.view(-1),
                                       ignore_index=-1)
            sop_loss = sop_loss.float()
            loss = lm_loss + sop_loss
            #averaged_losses = average_losses_across_data_parallel_group(
            #    [lm_loss, sop_loss])
            averaged_losses = [0, 0]
            return loss, {'lm loss': averaged_losses[0],
                          'sop loss': averaged_losses[1]}
        else:
            loss = lm_loss
            #averaged_losses = average_losses_across_data_parallel_group(
            #    [lm_loss])
            averaged_losses = [0]
            return loss, {'lm loss': averaged_losses[0]}

    tokens = torch.ones((micro_batch_size, seq_len)).cuda().long()
    padding_mask = \
        torch.ones(micro_batch_size, seq_len).cuda().bool()
    types = torch.ones((micro_batch_size, seq_len)).cuda().long()
    lm_labels = torch.ones((micro_batch_size, seq_len)).cuda().long()
    loss_mask = torch.ones((micro_batch_size, seq_len)).cuda().int()
    sentence_order = None

    def forward_step(data_iterator, model):
        if not args.bert_binary_head:
            types = None

        output_tensor = model(tokens, padding_mask, tokentype_ids=types,
                              lm_labels=lm_labels)
        return output_tensor, partial(loss_func, loss_mask, sentence_order)

    return model_provider, loss_func, forward_step


def benchmark_gpt_bert_one_case(benchmark_case):
    # Model configs
    model_type, batch_size, seq_len, hidden_size, num_layers, num_heads, \
        vocab_size, dp_size, tensor_mp_size, ddp_impl, checkpoint_activations \
        = benchmark_case

    # Parallel configs
    micro_batch_size = batch_size // dp_size

    # Initialize megatron
    sys.argv += ["--micro-batch-size", str(micro_batch_size)]
    sys.argv += ["--tensor-model-parallel-size", str(tensor_mp_size)]
    sys.argv += ["--global-batch-size", str(batch_size)]
    sys.argv += ["--num-layers", str(num_layers)]
    sys.argv += ["--hidden-size", str(hidden_size)]
    sys.argv += ["--num-attention-heads", str(num_heads)]
    sys.argv += ["--max-position-embeddings", str(seq_len)]
    sys.argv += ["--encoder-seq-length", str(seq_len)]
    sys.argv += ["--optimizer", "adam"]
    sys.argv += ["--train-iters", "100"]
    sys.argv += ["--lr", "0.00015"]
    sys.argv += ["--bert-no-binary-head"]
    sys.argv += ["--DDP-impl", "torch" if ddp_impl == 0 else "local"]
    sys.argv += ["--fp16"]
    if checkpoint_activations:
        sys.argv += ["--checkpoint-activations"]
    initialize_megatron()
    args = get_args()
    args.padded_vocab_size = vocab_size
    rank = torch.distributed.get_rank()

    # Check initialization
    assert dp_size == mpu.get_data_parallel_world_size()
    assert tensor_mp_size == mpu.get_tensor_model_parallel_world_size()

    # Build model
    if model_type == "gpt":
        model_provider, loss_func, forward_step = get_gpt_functions()
    elif model_type == "bert":
        model_provider, loss_func, forward_step = get_bert_functions()

    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)

    if rank == 0:
        parameter_count = compute_parameter_count(
            num_layers, hidden_size, vocab_size)
        print(model[0])
        print(f"Parameter count {parameter_count/1e9:.2f} B")

    # Benchmark step time
    def run_func():
        train_step(forward_step, None, model, optimizer, lr_scheduler)

    def sync_func():
        torch.cuda.synchronize()

    costs = benchmark_func(run_func, sync_func,
                           warmup=1, repeat=2, number=5)

    # Print results
    if rank == 0:
        peak_mem = torch.cuda.max_memory_allocated(0)
        tflops = compute_tflops(batch_size, seq_len, num_layers,
                                hidden_size, vocab_size,
                                torch.distributed.get_world_size(),
                                np.mean(costs))
        heads = ["Type", "Case", "Mesh Shape", "DDP Impl", "Checkpointing",
                 "Parameter Count", "Peak Mem", "Mean Time", "Std Time", "TFLOPS"]
        values = [model_type, str(benchmark_case[1:-4]),
                  str((dp_size, tensor_mp_size)),
                  "local" if ddp_impl else "torch",
                  str(checkpoint_activations),
                  f"{parameter_count/1e9:.3f}", f"{peak_mem/GB:5.3f}",
                  f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}", f"{tflops:.2f}"]
        write_tsv(heads, values, f"result_{model_type}.tsv")


if __name__ == "__main__":
    case = eval(sys.argv[-1])
    del sys.argv[-1]
    benchmark_gpt_bert_one_case(case)


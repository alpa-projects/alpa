import argparse
import gc
from functools import partial
import os
import sys
import time

import numpy as np

from megatron.utils import average_losses_across_data_parallel_group
from megatron.model import BertModel, GPTModel
from megatron.model import ModelType
from megatron import mpu, initialize_megatron, get_args, get_timers
from megatron.training import train_step, setup_model_and_optimizer
import torch

from util import write_tsv, benchmark_func,\
    compute_gpt_tflops, compute_gpt_parameter_count

GB = 1024**3


def get_gpt_functions():
    args = get_args()
    micro_batch_size = args.micro_batch_size
    seq_len = args.encoder_seq_length

    def model_provider(pre_process=True, post_process=True):
        model = GPTModel(num_tokentypes=0,
                         parallel_output=True,
                         pre_process=pre_process,
                         post_process=post_process)
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
        output_tensor = model(tokens,
                              position_ids,
                              attention_mask,
                              labels=labels)
        return output_tensor, partial(loss_func, loss_mask)

    return model_provider, loss_func, forward_step


def get_bert_functions():
    args = get_args()
    micro_batch_size = args.micro_batch_size
    seq_len = args.encoder_seq_length

    def model_provider(pre_process=True, post_process=True):
        num_tokentypes = 2 if args.bert_binary_head else 0
        model = BertModel(num_tokentypes=num_tokentypes,
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
            return loss, {
                'lm loss': averaged_losses[0],
                'sop loss': averaged_losses[1]
            }
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

        output_tensor = model(tokens,
                              padding_mask,
                              tokentype_ids=types,
                              lm_labels=lm_labels)
        return output_tensor, partial(loss_func, loss_mask, sentence_order)

    return model_provider, loss_func, forward_step


def benchmark_gpt_bert_one_case(benchmark_case, output_file_name):
    # Model configs
    model_type = "gpt"
    (global_batch_size, model_config, num_micro_batches, parallel_mode,
     parallel_args) = benchmark_case
    (seq_len, hidden_size, num_layers, num_heads,
     vocab_size) = model_config
    assert parallel_mode == "uniform"
    (prefer_reduce_scatter, use_remat, dp, op, pp,
     force_batch_dim_mapping) = parallel_args

    dp_size, tensor_mp_size, pipeline_mp_size = dp, op, pp
    checkpoint_activations = use_remat

    num_gpus = dp_size * tensor_mp_size * pipeline_mp_size
    assert global_batch_size % (dp_size * num_micro_batches) == 0
    micro_batch_size = global_batch_size // dp_size // num_micro_batches

    # always use local DDP
    ddp_impl = True

    # Parallel configs
    # Initialize megatron
    sys.argv += ["--micro-batch-size", str(micro_batch_size)]
    sys.argv += ["--tensor-model-parallel-size", str(tensor_mp_size)]
    sys.argv += ["--pipeline-model-parallel-size", str(pipeline_mp_size)]
    sys.argv += ["--global-batch-size", str(global_batch_size)]
    sys.argv += ["--num-layers", str(num_layers)]
    sys.argv += ["--hidden-size", str(hidden_size)]
    sys.argv += ["--num-attention-heads", str(num_heads)]
    sys.argv += ["--seq-length", str(seq_len)]
    sys.argv += ["--max-position-embeddings", str(seq_len)]
    sys.argv += ["--optimizer", "adam"]
    sys.argv += ["--train-iters", "100"]
    sys.argv += ["--lr", "0.00015"]
    sys.argv += ["--bert-no-binary-head"]
    sys.argv += ["--DDP-impl", "local" if ddp_impl else "torch"]
    sys.argv += ["--fp16"]
    sys.argv += ["--loss-scale", "8"]
    if checkpoint_activations:
        sys.argv += ["--checkpoint-activations"]
    # sys.argv += ["--no-masked-softmax-fusion"]
    # sys.argv += ["--no-async-tensor-model-parallel-allreduce"]
    # sys.argv += ["--no-scatter-gather-tensors-in-pipeline"]
    initialize_megatron()
    args = get_args()
    args.padded_vocab_size = vocab_size
    rank = torch.distributed.get_rank()

    # Check initialization
    assert dp_size == mpu.get_data_parallel_world_size()
    assert tensor_mp_size == mpu.get_tensor_model_parallel_world_size()
    assert pipeline_mp_size == mpu.get_pipeline_model_parallel_world_size()

    # Build model
    if model_type == "gpt":
        model_provider, loss_func, forward_step = get_gpt_functions()
    elif model_type == "bert":
        model_provider, loss_func, forward_step = get_bert_functions()

    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        model_provider, model_type=ModelType.encoder_or_decoder)

    parameter_count = compute_gpt_parameter_count(num_layers, hidden_size,
                                                  vocab_size)

    def run_func():
        train_step(forward_step, None, model, optimizer, lr_scheduler)

    # Warmup and reset timers
    run_func()
    timers = get_timers()
    names = list(timers.timers.keys())
    for name in names:
        timers(name).reset()

    # Benchmark step time
    repeat = 2
    number = 1
    costs = benchmark_func(run_func,
                           sync_func=None,
                           warmup=0,
                           repeat=repeat,
                           number=number)
    timers.log(names, normalizer=repeat * number)

    # Print results
    if rank == 0:
        peak_mem = torch.cuda.max_memory_allocated(0)
        tflops = compute_gpt_tflops(global_batch_size, seq_len, num_layers,
                                    hidden_size, vocab_size,
                                    torch.distributed.get_world_size(),
                                    np.mean(costs))
        tflops_ckpt = compute_gpt_tflops(global_batch_size, seq_len, num_layers,
                                         hidden_size, vocab_size,
                                         torch.distributed.get_world_size(),
                                         np.mean(costs), True)
        heads = [
            "Type", "Model Config", "Parallel Config", "P-mesh shape",
            "#Microbatch", "Force DP", "Remat", "Mean Time", "Std Time",
            "#Params", "TFLOPs", "TFLOPs (ckpt)", "Peak Mem"
        ]
        values = [
            model_type,
            str(benchmark_case[1:6]),
            str((dp_size, tensor_mp_size, pipeline_mp_size)), "N/A",
            str(num_micro_batches), "N/A",
            str(checkpoint_activations), f"{np.mean(costs):.3f}",
            f"{np.std(costs):.3f}", f"{parameter_count/1e9:.3f}",
            f"{tflops:.2f}", f"{tflops_ckpt:.2f}", f"{peak_mem/GB:5.3f}"
        ]
        write_tsv(heads, values,
                  f"{model_type}_megatron_{output_file_name}_rank{rank}.tsv")
        print("Sleeping for 30 seconds before starting the next case. ")
        time.sleep(30)


if __name__ == "__main__":
    case = eval(sys.argv[-2])
    output_file_name = sys.argv[-1]
    del sys.argv[-1]
    del sys.argv[-1]
    benchmark_gpt_bert_one_case(case, output_file_name)

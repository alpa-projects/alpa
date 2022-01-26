import time

import argparse
import os
import sys
import timeit
from functools import partial

import numpy as np

from benchmark.alpa.benchmark_gpt_bert import compute_tflops
from megatron.model.transformer import ParallelTransformer, ParallelMLP
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import ModelType
from megatron import mpu, initialize_megatron, get_args, get_timers
from megatron.training import train_step, setup_model_and_optimizer

import torch

from util import write_tsv, benchmark_func

GB = 1024 ** 3

# Note(Hao): in order for this to run with Megatron, disable the if-branch
# here in Megatron: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training.py#L390


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


def benchmark_transformer_layer_one_case(benchmark_case):
    # Model configs
    global_batch_size, seq_len, hidden_size, num_layers, num_heads, \
        dp_size, tensor_mp_size, pipeline_mp_size, num_micro_batches, \
        ddp_impl, checkpoint_activations = benchmark_case

    # Parallel configs
    assert global_batch_size % (dp_size * num_micro_batches) == 0
    micro_batch_size = global_batch_size // dp_size // num_micro_batches

    # Initialize megatron
    sys.argv += ["--micro-batch-size", str(micro_batch_size)]
    sys.argv += ["--tensor-model-parallel-size", str(tensor_mp_size)]
    sys.argv += ["--pipeline-model-parallel-size", str(pipeline_mp_size)]
    sys.argv += ["--global-batch-size", str(global_batch_size)]
    sys.argv += ["--num-layers", str(num_layers)]
    sys.argv += ["--hidden-size", str(hidden_size)]
    sys.argv += ["--num-attention-heads", str(num_heads)]
    sys.argv += ["--max-position-embeddings", str(seq_len)]
    sys.argv += ["--encoder-seq-length", str(seq_len)]
    sys.argv += ["--optimizer", "adam"]
    sys.argv += ["--train-iters", "100"]
    sys.argv += ["--lr", "0.00015"]
    sys.argv += ["--DDP-impl", "local" if ddp_impl else "torch"]
    # sys.argv += ["--no-scatter-gather-tensors-in-pipeline"]
    # sys.argv += ["--fp16"]
    if checkpoint_activations:
        sys.argv += ["--checkpoint-activations"]

    initialize_megatron()
    rank = torch.distributed.get_rank()

    # Check initialization
    assert dp_size == mpu.get_data_parallel_world_size()
    assert tensor_mp_size == mpu.get_tensor_model_parallel_world_size()
    assert pipeline_mp_size == mpu.get_pipeline_model_parallel_world_size()

    args = get_args()
    micro_batch_size = args.micro_batch_size
    seq_len = args.encoder_seq_length

    i = torch.cuda.current_device()
    x = torch.randn(seq_len, micro_batch_size, hidden_size).cuda(i)
    y = torch.randn(seq_len, micro_batch_size, hidden_size).cuda(i)
    attention_mask = torch.ones(micro_batch_size, 1, seq_len, seq_len). \
        to(torch.bool).cuda(i)


    def get_transformer_functions():
        args = get_args()

        def model_provider(pre_process=True, post_process=True):
            init_method_std = 0.02
            init_method = init_method_normal(init_method_std)
            scaled_init_method = scaled_init_method_normal(init_method_std, args.num_layers)
            model = ParallelTransformer(init_method, scaled_init_method, 0,
                                        pre_process=False, post_process=False)
            model.cuda(torch.cuda.current_device())
            return model

        def loss_func(output_tensor):
            loss = ((output_tensor - y) ** 2)
            loss = loss.mean()
            # averaged_losses = [0]
            return loss, {"avg loss": 0}

        def forward_step(data_iterator, model):
            # Note(Hao): Megatron PP uses model.module.input_tensor to overwrite
            # the input tensor to `model()`.
            if model.module.input_tensor == [None]:
                model.module.set_input_tensor(x)
            else:
                input_tensor = model.module.input_tensor
                model.module.set_input_tensor(input_tensor[0])
            output_tensor = model(x, attention_mask)
            return output_tensor, loss_func

        return model_provider, loss_func, forward_step

    # Build model
    model_provider, loss_func, forward_step = get_transformer_functions()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider,
                                                               model_type=ModelType.encoder_or_decoder)
    if rank == 0:
        print(model)

    def run_func():
        train_step(forward_step, None, model, optimizer, lr_scheduler)

    # Warmup and reset timers
    run_func()
    timers = get_timers()
    names = list(timers.timers.keys())
    for name in names:
        timers(name).reset()

    def sync_func():
        torch.cuda.synchronize()

    repeat = 10
    number = 1
    costs = benchmark_func(run_func, sync_func=sync_func,
                           warmup=0, repeat=repeat, number=number)
    timers.log(names, normalizer=repeat * number)


    # Print results
    # if rank == 0:
    peak_mem = torch.cuda.max_memory_allocated(0)
    heads = ["Type", "Case", "Mesh Shape", "#MB", "DDP Impl",
             "Peak Mem", "Mean Time", "Std Time"]
    values = ["transformer-layer", str(benchmark_case[:-3]),
              str(benchmark_case[-6:-3]), str(benchmark_case[-3]), str(benchmark_case[-2]),
              f"{peak_mem/GB:5.3f}", f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}", ]
    result_tsv = "result_trans-" + str(rank) + ".tsv"
    write_tsv(heads, values, result_tsv)
    time.sleep(10)


if __name__ == "__main__":
    case = eval(sys.argv[-1])
    del sys.argv[-1]
    benchmark_transformer_layer_one_case(case)


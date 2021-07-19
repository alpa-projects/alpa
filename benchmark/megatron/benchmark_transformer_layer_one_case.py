import argparse
import os
import sys
import timeit

import numpy as np
from megatron.model.transformer import ParallelTransformer, ParallelMLP
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from megatron.model import DistributedDataParallel as LocalDDP
from megatron import mpu, initialize_megatron, get_args
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

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


def benchmark_transformer_layer_one_case(benchmark_case):
    # Model configs
    batch_size, seq_len, hidden_size, num_layers, num_heads, \
        dp_size, tensor_mp_size, ddp_impl = benchmark_case

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
    initialize_megatron()
    rank = torch.distributed.get_rank()

    # Check initialization
    assert dp_size == mpu.get_data_parallel_world_size()
    assert tensor_mp_size == mpu.get_tensor_model_parallel_world_size()

    # Build model and input batch
    init_method_std = 0.02
    init_method = init_method_normal(init_method_std)
    scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
    model = ParallelTransformer(init_method, scaled_init_method, 0,
                                pre_process=False, post_process=False)
    model.cuda(torch.cuda.current_device())

    i = torch.cuda.current_device()
    if ddp_impl == 0:
        model = torchDDP(model, device_ids=[i], output_device=i,
                         process_group=mpu.get_data_parallel_group())
    elif ddp_impl == 1:
        model = LocalDDP(model, False, True)
    else:
        raise ValueError(f"Invalid ddp implementation: {ddp_impl}")

    if rank == 0:
        print(model)

    weight_mem = get_memory_usage() 

    x = torch.randn(seq_len, micro_batch_size, hidden_size).cuda(i)
    y = torch.randn(seq_len, micro_batch_size, hidden_size).cuda(i)
    attention_mask = torch.ones(micro_batch_size, 1, seq_len, seq_len).\
        to(torch.bool).cuda(i)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Benchmark step time
    def run_func():
        if isinstance(model, LocalDDP):
            model.zero_grad_buffer()
        else:
            optimizer.zero_grad()

        model.module.set_input_tensor(x)
        output = model(x, attention_mask)
        loss = ((output - y) ** 2)
        loss = loss.mean()

        loss.backward()

        if isinstance(model, LocalDDP):
            model.allreduce_gradients()
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    param.grad = param.main_grad

        optimizer.step()

    def sync_func():
        torch.cuda.synchronize()

    costs = benchmark_func(run_func, sync_func,
                           warmup=1, repeat=2, number=5)

    # Print results
    if rank == 0:
        peak_mem = torch.cuda.max_memory_allocated(0)
        heads = ["Type", "Case", "Mesh Shape", "DDP Impl", "Weight Mem",
                 "Peak Mem", "Mean Time", "Std Time"]
        values = ["transformer-layer", str(benchmark_case[:-3]),
                  str(benchmark_case[-3:-1]), str(benchmark_case[-1]),
                  f"{weight_mem/GB:5.3f}", f"{peak_mem/GB:5.3f}",
                  f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}"]
        write_tsv(heads, values, "result_trans.tsv")


if __name__ == "__main__":
    case = eval(sys.argv[-1])
    del sys.argv[-1]
    benchmark_transformer_layer_one_case(case)


"""
python3 -m torch.distributed.launch --nproc_per_node 4 benchmark_megatron.py
"""
import argparse
import os
import sys

import torch
from megatron.model.transformer import ParallelTransformerLayer, ParallelMLP
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from megatron import mpu, initialize_megatron

from timeit_v2 import py_benchmark
from timeit import timeit

MB = 1024 ** 2


def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    rank = torch.distributed.get_rank()
    device = rank % torch.cuda.device_count()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated


def benchmark_mlp():
    micro_batch_size = 4
    data_parallel_size = 1
    tensor_model_parallel_size = 4
    hidden_size = 2304
    num_attention_heads = 24
    seq_length = 512
    num_layers = 1
    init_method_std = 0.02
    init_method = init_method_normal(init_method_std)
    scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

    # Initialize megatron
    sys.argv += ["--micro-batch-size", str(micro_batch_size)]
    sys.argv += ["--tensor-model-parallel-size", str(tensor_model_parallel_size)]
    sys.argv += ["--global-batch-size", str(micro_batch_size)]
    sys.argv += ["--num-layers", str(num_layers)]
    sys.argv += ["--hidden-size", str(hidden_size)]
    sys.argv += ["--num-attention-heads", str(num_attention_heads)]
    sys.argv += ["--max-position-embeddings", str(seq_length)]
    sys.argv += ["--encoder-seq-length", str(seq_length)]
    initialize_megatron()
    rank = torch.distributed.get_rank()

    layer = ParallelMLP(init_method, scaled_init_method).cuda()

    weight_mem = get_memory_usage() 

    x = torch.randn(micro_batch_size, seq_length, hidden_size).cuda()
    y = torch.randn(micro_batch_size, seq_length, hidden_size).cuda()

    input_mem = get_memory_usage() - weight_mem
    before_backward_mem = [None]

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

    def func(record_peak=False):
        torch.distributed.barrier()
        optimizer.zero_grad()
        output = layer(x)[0]
        loss = ((output - y) ** 2)
        loss = loss.mean()
        if record_peak:
            before_backward_mem[0] = get_memory_usage()
        loss.backward()
        optimizer.step()
        torch.distributed.barrier()

    # Record peak memory
    func(True)
    func(True)
    before_backward_mem = before_backward_mem[0]

    # Benchmark time cost
    stmt = "func()"
    number = 100
    cost = timeit(stmt, globals={**globals(), **locals()},
                  number=number) / number

    # Print results
    if rank == 0:
        print(f"Weight mem: {weight_mem / MB:.2f} MB")
        print(f"Before backward mem: {before_backward_mem / MB:.2f} MB")
        print(f"Torch peak mem: {torch.cuda.max_memory_allocated(0) // MB:.2f} MB")
        print(f"Time: {cost * 1e3:.2f} ms")


def benchmark_transformer_layer():
    micro_batch_size = 4
    data_parallel_size = 1
    tensor_model_parallel_size = 4
    hidden_size = 2304
    num_attention_heads = 24
    seq_length = 512
    num_layers = 1
    init_method_std = 0.02
    init_method = init_method_normal(init_method_std)
    scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

    # Initialize megatron
    sys.argv += ["--micro-batch-size", str(micro_batch_size)]
    sys.argv += ["--tensor-model-parallel-size", str(tensor_model_parallel_size)]
    sys.argv += ["--global-batch-size", str(micro_batch_size)]
    sys.argv += ["--num-layers", str(num_layers)]
    sys.argv += ["--hidden-size", str(hidden_size)]
    sys.argv += ["--num-attention-heads", str(num_attention_heads)]
    sys.argv += ["--max-position-embeddings", str(seq_length)]
    sys.argv += ["--encoder-seq-length", str(seq_length)]
    initialize_megatron()
    rank = torch.distributed.get_rank()

    layer = ParallelTransformerLayer(init_method, scaled_init_method, 0).cuda()

    weight_mem = get_memory_usage() 

    x = torch.randn(micro_batch_size, seq_length, hidden_size).cuda()
    y = torch.randn(micro_batch_size, seq_length, hidden_size).cuda()
    attention_mask = torch.ones(seq_length,
                                num_attention_heads // tensor_model_parallel_size,
                                micro_batch_size,
                                micro_batch_size).to(torch.bool).cuda()

    input_mem = get_memory_usage() - weight_mem
    before_backward_mem = [None]

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

    def func(record_peak=False):
        torch.distributed.barrier()
        optimizer.zero_grad()
        output = layer(x, attention_mask)
        loss = ((output - y) ** 2)
        loss = loss.mean()
        if record_peak:
            before_backward_mem[0] = get_memory_usage()
        loss.backward()
        optimizer.step()
        torch.distributed.barrier()

    # Record peak memory
    func(True)
    func(True)
    before_backward_mem = before_backward_mem[0]

    # Benchmark time cost
    stmt = "func()"
    number = 100
    cost = timeit(stmt, globals={**globals(), **locals()},
                  number=number) / number

    # Print results
    if rank == 0:
        print(f"Weight mem: {weight_mem / MB:.2f} MB")
        print(f"Before backward mem: {before_backward_mem / MB:.2f} MB")
        print(f"Torch peak mem: {torch.cuda.max_memory_allocated(0) // MB:.2f} MB")
        print(f"Time: {cost * 1e3:.2f} ms")


if __name__ == "__main__":
    #benchmark_mlp()
    benchmark_transformer_layer()


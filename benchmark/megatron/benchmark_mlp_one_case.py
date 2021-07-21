import argparse
import os
import sys

import numpy as np
from megatron.model.transformer import ParallelTransformerLayer, ParallelMLP
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
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated


class MultiLayerMLP(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()

        self.num_layers = num_layers

        init_method_std = 0.02
        init_method = init_method_normal(init_method_std)
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
        for i in range(self.num_layers):
            setattr(self, f"layer_{i}", ParallelMLP(init_method, scaled_init_method))

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            out, out_bias = getattr(self, f"layer_{i}")(out)
            out = out + out_bias
        return out


def benchmark_mlp_one_case(benchmark_case):
    # Model configs
    batch_size, seq_len, hidden_size, num_layers, num_heads, \
        dp_size, tensor_mp_size, ddp_impl = benchmark_case

    # Parallel configs
    micro_batch_size = batch_size // dp_size

    # Initialize megatron
    sys.argv += ["--micro-batch-size", str(micro_batch_size)]
    sys.argv += ["--tensor-model-parallel-size", str(tensor_mp_size)]
    sys.argv += ["--global-batch-size", str(micro_batch_size * dp_size)]
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
    model = MultiLayerMLP(num_layers)
    model.cuda(torch.cuda.current_device())

    i = torch.cuda.current_device()
    if ddp_impl == 0:
        model = torchDDP(model, device_ids=[i], output_device=i,
                         process_group=mpu.get_data_parallel_group())
    else:
        model = LocalDDP(model, False, True)

    if rank == 0:
        print(model)

    weight_mem = get_memory_usage() 

    x = torch.randn(micro_batch_size, seq_len, hidden_size).cuda()
    y = torch.randn(micro_batch_size, seq_len, hidden_size).cuda()

    input_mem = get_memory_usage() - weight_mem
    before_backward_mem = [None]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Benchmark step time
    def run_func():
        if isinstance(model, LocalDDP):
            model.zero_grad_buffer()
        else:
            optimizer.zero_grad()

        output = model(x)
        loss = ((output - y) ** 2)
        loss = loss.mean()
        loss.backward()

        if isinstance(model, LocalDDP):
            model.allreduce_gradients()
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    param.grad = param.main_grad

        optimizer.step()

        torch.distributed.barrier()

    def sync_func():
        torch.cuda.synchronize()

    costs = benchmark_func(run_func, sync_func,
                           warmup=1, repeat=2, number=5)

    # Print results
    if rank == 0:
        peak_mem = torch.cuda.max_memory_allocated(0)
        heads = ["Type", "Case", "WeightMem", "PeakMem", "Mean Time", "Std Time"]
        values = ["mlp", str(benchmark_case), f"{weight_mem/GB:.2f}", f"{peak_mem/GB:.2f}",
                  f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}"]
        write_tsv(heads, values, "result_mlp.tsv")


if __name__ == "__main__":
    case = eval(sys.argv[-1])
    del sys.argv[-1]
    benchmark_mlp_one_case(case)


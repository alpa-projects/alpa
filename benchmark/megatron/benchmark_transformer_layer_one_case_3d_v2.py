import argparse
import os
import sys
import timeit
from functools import partial

import numpy as np
from torch.utils.data import Dataset, DataLoader

from megatron.model.transformer import ParallelTransformer, ParallelMLP
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from megatron.model import DistributedDataParallel as LocalDDP
from megatron import mpu, initialize_megatron, get_args, get_timers
from megatron.training import train_step, setup_model_and_optimizer

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
    # sys.argv += ["--bert-no-binary-head"]
    sys.argv += ["--DDP-impl", "local" if ddp_impl else "torch"]
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


    # class DummyDataset(Dataset):
    #     def __len__(self):
    #         return 1000
    #
    #     def __getitem__(self, idx):
    #         return x, y

    # my_dataset = DummyDataset()
    # data_iter = iter(DataLoader(my_dataset))

    # def get_batch(data_iterator):
    #     x, y = next(data_iterator)
    #     return x, y

    def get_transformer_functions():
        args = get_args()

        def model_provider(pre_process=True, post_process=True):
            init_method_std = 0.02
            init_method = init_method_normal(init_method_std)
            scaled_init_method = scaled_init_method_normal(init_method_std, args.num_layers)
            model = ParallelTransformer(init_method, scaled_init_method, 0,
                                        pre_process=False, post_process=False)
            # model.cuda(torch.cuda.current_device())
            return model

        def loss_func(output_tensor):
            loss = ((output_tensor - y) ** 2)
            loss = loss.mean()
            averaged_losses = [0]
            return loss, {"avg loss": 0}

        def forward_step(data_iterator, model):
            # x, y = get_batch(data_iterator)
            # print(x.shape)
            # print(y.shape)
            # model.set_input_tensaor
            if model.module.input_tensor is None:
                model.module.set_input_tensor(x)
            output_tensor = model(x, attention_mask)
            return output_tensor, loss_func

        return model_provider, loss_func, forward_step



    # Build model
    model_provider, loss_func, forward_step = get_transformer_functions()


    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    if rank == 0:
        print(model)
    # print(model)
    # print(optimizer)
    # print(lr_scheduler)
    # # Build model and input batch
    # init_method_std = 0.02
    # init_method = init_method_normal(init_method_std)
    # scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
    # model = ParallelTransformer(init_method, scaled_init_method, 0,
    #                             pre_process=False, post_process=False)
    # model.cuda(torch.cuda.current_device())
    #
    # i = torch.cuda.current_device()
    # if ddp_impl == 0:
    #     model = torchDDP(model, device_ids=[i], output_device=i,
    #                      process_group=mpu.get_data_parallel_group())
    # elif ddp_impl == 1:
    #     model = LocalDDP(model, False, True)
    # else:
    #     raise ValueError(f"Invalid ddp implementation: {ddp_impl}")
    #
    # if rank == 0:
    #     print(model)
    #
    # weight_mem = get_memory_usage()
    #
    # x = torch.randn(seq_len, micro_batch_size, hidden_size).cuda(i)
    # y = torch.randn(seq_len, micro_batch_size, hidden_size).cuda(i)
    # attention_mask = torch.ones(micro_batch_size, 1, seq_len, seq_len).\
    #     to(torch.bool).cuda(i)
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    #
    # # Benchmark step time
    # def run_func():
    #     if isinstance(model, LocalDDP):
    #         model.zero_grad_buffer()
    #     else:
    #         optimizer.zero_grad()
    #
    #     model.module.set_input_tensor(x)
    #     output = model(x, attention_mask)
    #     loss = ((output - y) ** 2)
    #     loss = loss.mean()
    #
    #     loss.backward()
    #
    #     if isinstance(model, LocalDDP):
    #         model.allreduce_gradients()
    #         for param_group in optimizer.param_groups:
    #             for param in param_group['params']:
    #                 param.grad = param.main_grad
    #
    #     optimizer.step()

    def run_func():
        train_step(forward_step, None, model, optimizer, lr_scheduler)

    # Warmup and reset timers
    run_func()
    if rank == 0:
        print(">>>>running in v2....\n")
    timers = get_timers()
    names = list(timers.timers.keys())
    for name in names:
        timers(name).reset()

    def sync_func():
        torch.cuda.synchronize()

    repeat = 10
    number = 1
    costs = benchmark_func(run_func, sync_func=sync_func,
                           warmup=1, repeat=repeat, number=number)

    # Print results
    if rank == 0:
        peak_mem = torch.cuda.max_memory_allocated(0)
        heads = ["Type", "Case", "Mesh Shape", "DDP Impl", "Weight Mem",
                 "Peak Mem", "Mean Time", "Std Time"]
        values = ["transformer-layer", str(benchmark_case[:-3]),
                  str(benchmark_case[-6:-3]), str(benchmark_case[-2]),
                  f"{0/GB:5.3f}", f"{peak_mem/GB:5.3f}",
                  f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}"]
        write_tsv(heads, values, "result_trans.tsv")


if __name__ == "__main__":
    case = eval(sys.argv[-1])
    del sys.argv[-1]
    # case = (32,  1024, 1536, 2,  1536//96,  2, 1, 2, 1, 1, 0)
    benchmark_transformer_layer_one_case(case)


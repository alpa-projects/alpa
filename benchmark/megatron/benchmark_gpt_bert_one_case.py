import argparse
import gc
import os
import sys
import timeit


import numpy as np
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from megatron.model import BertModel, GPTModel, Float16Module, DistributedDataParallel as LocalDDP
from megatron import mpu, initialize_megatron, get_args
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP


from timeit_v2 import py_benchmark
from benchmark_mlp_one_case import write_tsv

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


def benchmark_gpt_bert_one_case(benchmark_case):
    # Model configs
    model_type, batch_size, seq_len, hidden_size, num_layers, num_heads, \
        vocab_size, dp_size, tensor_mp_size, ddp_impl = benchmark_case

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
    #sys.argv += ["--checkpoint-activations"]
    #sys.argv += ["--fp16"]
    initialize_megatron()
    get_args().padded_vocab_size = vocab_size
    rank = torch.distributed.get_rank()

    # Check initialization
    assert dp_size == mpu.get_data_parallel_world_size()
    assert tensor_mp_size == mpu.get_tensor_model_parallel_world_size()

    # Build model and input batch
    init_method_std = 0.02
    init_method = init_method_normal(init_method_std)
    scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
    if model_type == "bert":
        model = BertModel(add_binary_head=False)
    elif model_type == "gpt":
        model = GPTModel()
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    model.cuda(torch.cuda.current_device())

    args = get_args()
    if args.fp16:
        model = Float16Module(model, args)

    weight_mem = get_memory_usage()

    i = torch.cuda.current_device()
    if ddp_impl == 0:
        model = torchDDP(model, device_ids=[i], output_device=i,
                         process_group=mpu.get_data_parallel_group())
    elif ddp_impl == 1:
        use_contiguous_buffers_in_ddp = False
        model = LocalDDP(model, False, use_contiguous_buffers_in_ddp)
    else:
        raise ValueError(f"Invalid ddp implementation: {ddp_impl}")

    weight_size = 0
    for p in model.parameters():
        weight_size += np.prod(p.shape)

    if rank == 0:
        print(model)
        print(f"Weight mem {weight_mem/GB:.2f} GB, " +
              f"Weight size {weight_size/GB:.2f} B")

    input_ids = torch.ones((micro_batch_size, seq_len)).cuda(i).long()
    position_ids = torch.ones((micro_batch_size, seq_len)).cuda(i).long()
    tokentype_ids = torch.ones((micro_batch_size, seq_len)).cuda(i).long()
    lm_labels = torch.ones((micro_batch_size, seq_len)).cuda(i).long()
    if model_type == "bert":
        attention_mask = \
            torch.ones(micro_batch_size, seq_len).cuda().long()
    elif model_type == "gpt":
        attention_mask = \
            torch.ones(micro_batch_size, 1, seq_len, seq_len).cuda().bool()

    input_mem = get_memory_usage() - weight_mem
    act_mem = [None]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    def func(record_act_mem=False):
        if isinstance(model, LocalDDP) and use_contiguous_buffers_in_ddp:
            model.zero_grad_buffer()
        else:
            optimizer.zero_grad()

        if model_type == "bert":
            lm_loss, binary_logits = model(input_ids, attention_mask,
                                           tokentype_ids, lm_labels)
        elif model_type == "gpt":
            lm_loss = model(input_ids, position_ids, attention_mask, lm_labels)

        loss = lm_loss.mean()

        if record_act_mem:
            before_backward_mem = get_memory_usage()
            loss.backward()
            act_mem[0] = before_backward_mem - get_memory_usage()
        else:
            loss.backward()

        if isinstance(model, LocalDDP):
            model.allreduce_gradients()
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    param.grad = param.main_grad

        optimizer.step()

        torch.distributed.barrier()

    # Record peak memory
    func(True)
    func(True)
    peak_mem = torch.cuda.max_memory_allocated(0)

    # Benchmark time cost
    stmt = "func()"
    repeat = 3
    number = 3
    costs = np.array(timeit.repeat(stmt, globals={**globals(), **locals()},
        repeat=repeat, number=number)) / number
    total_flop = 72 * batch_size * seq_len * (hidden_size ** 2) * num_layers * \
      (1 + seq_len / (6 * hidden_size)) \
      + 6 * batch_size * seq_len * hidden_size * vocab_size
    tflops = total_flop / np.mean(costs) / torch.distributed.get_world_size() / 1e12

    # Print results
    if rank == 0:
        heads = ["Type", "Case", "Mesh Shape", "DDP Impl", "Weight Mem",
                 "Peak Mem", "Mean Time", "Std Time", "TFLOPS"]
        values = [model_type, str(benchmark_case[1:-3]),
                  str(benchmark_case[-3:-1]), str(benchmark_case[-1]),
                  f"{weight_mem/GB:5.3f}", f"{peak_mem/GB:5.3f}",
                  f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}",
                  f"{tflops:.2f}"]
        write_tsv(heads, values, f"result_{model_type}.tsv")


if __name__ == "__main__":
    case = eval(sys.argv[-1])
    del sys.argv[-1]
    benchmark_gpt_bert_one_case(case)


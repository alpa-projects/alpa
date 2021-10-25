import os
import time

import numpy as np

def write_tsv(heads, values, filename, print_line=True):
    """Write tsv data to a file."""
    assert len(heads) == len(values)

    values = [str(x) for x in values]

    with open(filename, "a") as fout:
        fout.write("\t".join(values) + "\n")

    if print_line:
        line = ""
        for i in range(len(heads)):
            line += heads[i] + ": " + values[i] + "  "
        print(line)


def benchmark_func(run_func, sync_func=None, warmup=1, repeat=3, number=5):
    """Benchmark the execution time of a function."""
    costs = []

    # Warmup
    for i in range(warmup):
        run_func()

    # Benchmark
    for i in range(repeat):
        if sync_func:
            sync_func()
        tic = time.time()

        for j in range(number):
            run_func()

        if sync_func:
            sync_func()
        costs.append(time.time() - tic)

    return np.array(costs) / number


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


def get_torch_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    import torch
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if print_info:
        print("allocated: %.2f GB" % (allocated / GB), flush=True)
        print("reserved:  %.2f GB" % (reserved / GB), flush=True)
    return allocated


def compute_gpt_tflops(batch_size, seq_len, num_layers, hidden_size, vocab_size,
                       num_gpus, latency, checkpoint_activations=False):
    factor = 96 if checkpoint_activations else 72
    total_flop = factor * batch_size * seq_len * (hidden_size ** 2) * num_layers * \
          (1 + seq_len / (6 * hidden_size)) \
          + 6 * batch_size * seq_len * hidden_size * vocab_size
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops


def compute_gpt_parameter_count(num_layers, hidden_size, vocab_size):
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


GB = 1 << 30

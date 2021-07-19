import os
import time

import numpy as np

def write_tsv(heads, values, filename, print_line=True):
    """Write tsv data to a file."""
    assert len(heads) == len(values)

    with open(filename, "a") as fout:
        fout.write("\t".join(values) + "\n")

    if print_line:
        line = ""
        for i in range(len(heads)):
            line += heads[i] + ": " + values[i] + "  "
        print(line)


def benchmark_func(run_func, sync_func, warmup=1, repeat=3, number=5):
    """Benchmark the execution time of a function."""
    costs = []

    # Warmup
    for i in range(warmup):
        run_func()
    sync_func()

    # Benchmark
    for i in range(repeat):
        tic = time.time()
        for j in range(number):
            run_func()
        sync_func()
        costs.append(time.time() - tic)

    return np.array(costs) / number


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


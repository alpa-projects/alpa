import argparse
import dataclasses
from functools import partial
import os
import time

import jax
from jax.tree_util import tree_flatten, tree_leaves
import numpy as np
import alpa
from alpa.testing import assert_allclose
from alpa.util import compute_bytes

from examples.opt_serving.model.opt_model import (get_config,
                                                  get_pipeshard_executable,
                                                  load_params_dis_array,
                                                  init_cache_dis_array)

GB = 1 << 30


def test_load(name, dummy, batch_size):
    config = get_config(name, num_pp_stages=2)
    path = f"/home/ubuntu/opt_weights/{name}_np"

    alpa.init()

    input_ids = np.array([[5625, 16, 10, 2721, 183, 8, 38, 236, 7]],
                         dtype=np.int32)
    input_ids = np.tile(input_ids, [batch_size, 1])
    print("input_ids", input_ids)

    print("Compile...")
    tic = time.time()
    executable, params_aval = get_pipeshard_executable(config)
    params_info, _ = executable.get_load_info()
    executable.sync()
    print(f"Duration: {time.time() - tic:.2f}")

    print("Load model...")
    tic = time.time()
    params = load_params_dis_array(path,
                                   executable,
                                   params_aval,
                                   config,
                                   dummy=dummy)
    executable.sync()
    duration = time.time() - tic
    num_bytes = compute_bytes(params)
    print(
        f"Duration: {duration:.2f}, Bandwidth: {num_bytes / duration / GB:.2f} GB/s"
    )

    print("Build cache...")
    tic = time.time()
    init_cache = init_cache_dis_array(executable,
                                      config,
                                      batch_size,
                                      dummy=dummy)
    executable.sync()
    duration = time.time() - tic
    num_bytes = compute_bytes(init_cache)
    print(
        f"Duration: {duration:.2f}, Bandwidth: {num_bytes / duration / GB:.2f} GB/s"
    )

    # Run
    for _ in range(2):
        start_time = time.time()
        cache = init_cache
        for i in range(input_ids.shape[1]):
            tic = time.time()
            input_ids_step = input_ids[:, i:i + 1]
            position_ids_step = np.full_like(input_ids_step, i + config.pad + 1)
            output = executable(
                params, {
                    "input_ids": input_ids_step,
                    "position_ids": position_ids_step,
                    "cache": cache,
                })
            cache = output.attention_cache

            logits_step = str(output.logits).replace("\n", " ")
            print(
                f"step: {i}, latency: {time.time() - tic:.2f}, "
                f"exec_cost: {executable.get_execution_time_costs(warmup=0)[-1]:.2f}, "
                f"logits: {logits_step}")
        duration = time.time() - start_time
        latency = duration / np.prod(input_ids.shape)
        print(
            f"latency: {latency:.2f} s/token, throughput: {1/latency:.2f} token/s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="125M")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    test_load(args.model, args.dummy, args.batch_size)

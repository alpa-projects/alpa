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

from opt_model import (get_config, init_model_aval,
                       build_init_cache, build_position_ids, load_np_params,
                       get_pipeshard_executable)

GB = 1 << 30


def test_load(name):
    config = get_config(name, num_pp_stages=2)
    np_weights_folder = f"/home/ubuntu/opt_weights/{name}_np"
    ts_weights_folder = f"/home/ubuntu/opt_weights/{name}_ts"
    use_ts = False

    alpa.init()

    input_ids = np.array([[5625,   16,   10, 2721,  183,    8,   38,  236,    7]], dtype=np.int32)
    position_ids = build_position_ids(input_ids, config.pad)
    print("input_ids", input_ids)

    print("Compile...")
    tic = time.time()
    executable, params_aval = get_pipeshard_executable(config)
    params_info, _ = executable.get_load_info()
    executable.sync()
    print(f"Duration: {time.time() - tic:.2f}")

    print("Load model...")
    tic = time.time()
    if use_ts:
        params = alpa.restore_checkpoint(ts_weights_folder, 1, params_info, params_info)
    else:
        dummy = False
        alpa.global_config.use_dummy_value_for_benchmarking = dummy
        params = load_np_params(params_aval, np_weights_folder, config, dummy=dummy)
        flat_args, in_tree = tree_flatten(params)
        flat_info = tree_leaves(params_info)
        params = executable.mesh_group.shard_args_to_arrays(flat_info, flat_args)
    executable.sync()
    duration = time.time() - tic
    num_bytes = compute_bytes(params)
    print(f"Duration: {duration:.2f}, Bandwidth: {num_bytes / duration / GB:.2f} GB/s")

    print("Build cache...")
    tic = time.time()
    cache = build_init_cache(config)
    _, batch_info = executable.get_load_info()
    cache_info = batch_info["cache"]
    flat_args, in_tree = tree_flatten(cache)
    flat_info = tree_leaves(cache_info)
    cache = executable.mesh_group.shard_args_to_arrays(flat_info, flat_args)
    duration = time.time() - tic
    num_bytes = compute_bytes(cache)
    print(f"Duration: {duration:.2f}, Bandwidth: {num_bytes / duration / GB:.2f} GB/s")

    # Run
    for i in range(input_ids.shape[1]):
        tic = time.time()
        input_ids_step = input_ids[:, i:i+1]
        position_ids_step = np.full_like(input_ids_step, i + config.pad + 1)
        logits_step, cache = executable(params, {
            "input_ids": input_ids_step,
            "position_ids": position_ids_step,
            "cache": cache,
        })

        logits_step = str(logits_step).replace("\n", " ")
        print(f"step: {i}, latency: {time.time() - tic:.2f}, "
              f"exec_cost: {executable.get_execution_time_costs(warmup=0)[-1]:.2f}, "
              f"logits: {logits_step}")


if __name__ == "__main__":
    name = "125M"

    test_load(name)

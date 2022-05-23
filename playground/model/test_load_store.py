import dataclasses
from functools import partial
import os

import jax
import jax.numpy as jnp
import numpy as np
from alpa.testing import assert_allclose
import alpa

from opt_model import (OPTConfig, OPTForLMModule, get_config,
                       init_model_aval, inference_step_no_cache,
                       build_init_cache, build_init_cache_aval,
                       build_position_ids, load_np_params,
                       get_pipeshard_executable)

def test_save_numpy_to_ts():
    name = "125M"
    config = get_config(name, num_pp_stages=2)
    numpy_weights_folder = os.path.abspath(f"./{name}_numpy_weights")
    ckpt_dir = os.path.abspath(f"{name}_ts_weights")

    alpa.init()

    # Load model
    executable, params_aval = get_pipeshard_executable(config)
    params_info, _ = executable.get_load_info()
    params = load_np_params(params_aval, numpy_weights_folder, config)

    # Save model
    alpa.save_checkpoint(ckpt_dir, params, 1)


def test_load_ts():
    name = "125M"
    config = get_config(name, num_pp_stages=2)
    ckpt_dir = os.path.abspath(f"{name}_ts_weights")

    alpa.init()

    input_ids = jnp.array([[5625,   16,   10, 2721,  183,    8,   38,  236,    7]], dtype=jnp.int32)
    position_ids = build_position_ids(input_ids, config.pad)
    print("input_ids", input_ids)

    # Load model
    executable, _ = get_pipeshard_executable(config)
    params_info, _ = executable.get_load_info()
    params = alpa.restore_checkpoint(ckpt_dir, 1, params_info, params_info)

    cache = build_init_cache(config)
    for i in range(input_ids.shape[1]):
        input_ids_step = input_ids[:, i:i+1]
        position_ids_step = jnp.full_like(input_ids_step, i + config.pad + 1)
        logits_step, cache = executable(params, {
            "input_ids": input_ids_step,
            "position_ids": position_ids_step,
            "cache": cache,
        })
        print(i, logits_step)

if __name__ == "__main__":
    #test_save_numpy_to_ts()
    test_load_ts()

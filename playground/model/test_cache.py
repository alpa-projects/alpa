from functools import partial
import os

import jax
import jax.numpy as jnp
import numpy as np
from alpa.testing import assert_allclose

from opt_model import (OPTConfig, OPTForLMModule, get_config,
                       init_model_aval, inference_step_no_cache,
                       build_init_cache, build_position_ids, load_np_params)


def print_params(params, prefix=""):
    for key, value in params.items():
        if isinstance(value, dict):
            print_params(value, prefix=prefix + key + ".")
        else:
            print(prefix + key, value.shape)


def test_opt_125M():
    #TODO: align dtype
    config = get_config("125M")
    numpy_weights_folder = "./numpy_weights"

    # Init model
    input_ids = jnp.array([[5625,   16,   10, 2721,  183,    8,   38,  236,    7]], dtype=jnp.int32)
    position_ids = build_position_ids(input_ids, config.pad)
    print("input_ids", input_ids)

    model, params = init_model_aval(config)
    params = load_np_params(params.unfreeze(), numpy_weights_folder, num_layers=config.decoder_layers)

    # Get expected results
    logits_no_cache = inference_step_no_cache(params, {
        "input_ids": input_ids,
        "position_ids": position_ids,
    }, model.apply)

    print("logits_no_cache", logits_no_cache)

    # JIT
    @partial(jax.jit, static_argnums=(2,))
    def inference_step_with_cache(params, batch, apply_func):
        print("traced")
        output = apply_func(params,
                            batch["input_ids"],
                            batch["position_ids"],
                            attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    cache = build_init_cache(config)

    for i in range(input_ids.shape[1]):
        input_ids_step = input_ids[:, i:i+1]
        position_ids_step = jnp.full_like(input_ids_step, i + config.pad + 1)
        logits_step, cache = inference_step_with_cache(params, {
            "input_ids": input_ids_step,
            "position_ids": position_ids_step,
            "cache": cache,
        }, model.apply)
        assert_allclose(logits_step, logits_no_cache[:, i:i+1])


if __name__ == "__main__":
    test_opt_125M()

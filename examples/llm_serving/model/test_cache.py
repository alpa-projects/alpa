"""Test the correctness of cache implementation."""
import jax
import jax.numpy as jnp
import numpy as np

from alpa.testing import assert_allclose
from llm_serving.model.opt_model import (get_opt_config, init_model_aval,
                                         inference_step_no_cache,
                                         init_cache_np,
                                         build_position_ids,
                                         load_params_np)


def print_params(params, prefix=""):
    for key, value in params.items():
        if isinstance(value, dict):
            print_params(value, prefix=prefix + key + ".")
        else:
            print(prefix + key, value.shape)


def test_opt_125M(decompose_input):
    print("Testing cache with decompose_input=%s" % decompose_input)
    name = "125M"
    config = get_opt_config(name, dtype=jnp.float32)
    np_weights_folder = f"/home/ubuntu/opt_weights/{name}_np"
    batch_size = 1

    # Init model
    input_ids = np.array([[5625, 16, 10, 2721, 183, 8, 38, 236, 7]],
                         dtype=np.int32)
    input_ids = np.tile(input_ids, [batch_size, 1])
    position_ids = build_position_ids(input_ids, config.pad)
    print("input_ids", input_ids)

    model, params = init_model_aval(config)
    params = load_params_np(params, np_weights_folder, config)
    params = jax.tree_map(jnp.array, params)

    # Get expected results
    logits_no_cache = inference_step_no_cache(params, {
        "input_ids": input_ids,
        "position_ids": position_ids,
    }, model.apply)
    print("logits_no_cache", logits_no_cache)

    # JIT
    @jax.jit
    def inference_step_with_cache(params, batch):
        print("traced")
        output = model.apply(params,
                             batch["input_ids"],
                             batch["position_ids"],
                             attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    cache = init_cache_np(config, input_ids.shape[0])

    if decompose_input:
        # Decompose input so that all input lengths are one.
        for i in range(input_ids.shape[1]):
            input_ids_step = input_ids[:, i:i + 1]
            position_ids_step = np.full_like(input_ids_step, i + config.pad + 1)
            logits_step, cache = inference_step_with_cache(
                params, {
                    "input_ids": input_ids_step,
                    "position_ids": position_ids_step,
                    "cache": cache
                })
            assert_allclose(logits_step, logits_no_cache[:, i:i + 1])
    else:
        # Same as inference_step_no_cache that has input length > 1.
        logits_step, cache = inference_step_with_cache(
            params, {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "cache": cache
            })
        assert_allclose(logits_step, logits_no_cache)


if __name__ == "__main__":
    test_opt_125M(False)
    test_opt_125M(True)

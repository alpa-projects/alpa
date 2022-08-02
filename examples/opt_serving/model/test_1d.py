"""Test the correctness of cache implementation."""
import jax
import jax.numpy as jnp
import numpy as np

from alpa.testing import assert_allclose
from opt_serving.model import opt_model
from opt_serving.model import opt_model_1d


def print_params(params, prefix=""):
    for key, value in params.items():
        if isinstance(value, dict):
            print_params(value, prefix=prefix + key + ".")
        else:
            print(prefix + key, value.shape)


def inference_step_with_cache(model):

    @jax.jit
    def wrapper(params, batch):
        output = model.apply(params,
                             batch["input_ids"],
                             batch["position_ids"],
                             attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    return wrapper


def test_opt_125M():
    name = "125M"
    np_weights_folder = f"/home/ubuntu/opt_weights/{name}_np"

    input_id_list = [
        [45942, 2866, 16, 5, 892, 9, 44042, 8],
        # [100, 261, 23888, 2426, 16, 10, 21624, 12, 4310, 3034, 9744, 25526, 11],
        # [133, 589, 9, 886, 6, 10817, 16, 10, 285],
        # [5625, 16, 10, 205, 183, 8, 38, 236, 7],
        # [2264, 16, 5, 7440, 9, 16673, 873, 24214, 116],
    ]
    batch_size = len(input_id_list)

    # Init 2D model
    print("Running 2D OPT model", flush=True)
    config = opt_model.get_opt_config(name, dtype=jnp.float32)
    model_2d, params_2d = opt_model.init_model_aval(config)
    params_2d = opt_model.load_params_np(params_2d, np_weights_folder, config)
    params_2d = jax.tree_map(jnp.array, params_2d)
    cache_2d = opt_model.init_cache_np(config, 1)

    # Get expected results
    @jax.jit
    def inference_step_2d(params, batch):
        output = model_2d.apply(params,
                                batch["input_ids"],
                                batch["position_ids"],
                                attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    # Run each prompt individually because their lengths are different
    logits_ref = []
    for input_ids in input_id_list:
        input_ids = np.array([input_ids], dtype=np.int32)
        position_ids = opt_model.build_position_ids(input_ids, config.pad)

        # Note: do not override the cache
        logits, temp = inference_step_2d(
            params_2d, {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "cache": cache_2d,
            })
        # The logits are in shape (1, seq_len, vocab_size), so we squeeze the batch dimension
        # and concat them together to mimic the output of the 1D model.
        logits_ref.append(logits.squeeze())
    logits_ref = np.concatenate(logits_ref, axis=0)
    print("logits_ref shape:", logits_ref.shape, flush=True)
    for i, t in enumerate(temp):
        jnp.save("temp_2d_%d.npy" % i, t)

    # Init 1D model
    print("Running 1D OPT model", flush=True)
    model_1d, params_1d = opt_model_1d.init_model_aval(config, batch_size)
    params_1d = opt_model_1d.load_params_np(params_1d, np_weights_folder,
                                            config)
    params_1d = jax.tree_map(jnp.array, params_1d)
    cache_1d = opt_model_1d.init_cache_np(config, batch_size)

    @jax.jit
    def inference_step_1d(params, batch):
        output = model_1d.apply(params,
                                batch["input_ids"],
                                batch["position_ids"],
                                batch["batch_idxs"],
                                attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    # Concat promopts together
    input_ids_flatten = []
    position_ids_flatten = []
    batch_idxs = []
    for idx, input_ids in enumerate(input_id_list):
        # fused_mmha kernel requires batch_idxs to be in shape sum(input_ids). Each element
        # in batch_idxs is the ID of the corresponding input sequence (starting from 1).
        batch_idxs += np.full(shape=(len(input_ids),),
                              fill_value=idx + 1,
                              dtype=np.int32).tolist()
        input_ids_flatten += input_ids
        position_ids = opt_model_1d.build_position_ids(
            np.array(input_ids, dtype=np.int32), config.pad).tolist()
        position_ids_flatten += position_ids
    input_ids_flatten = np.array(input_ids_flatten, dtype=np.int32)
    position_ids_flatten = np.array(position_ids_flatten, dtype=np.int32)
    batch_idxs = np.array(batch_idxs, dtype=np.int32)

    logits, temp = inference_step_1d(
        params_1d, {
            "input_ids": input_ids_flatten,
            "position_ids": position_ids_flatten,
            "batch_idxs": batch_idxs,
            "cache": cache_1d,
        })
    print("logits shape:", logits.shape, flush=True)
    for i, t in enumerate(temp):
        jnp.save("temp_1d_%d.npy" % i, t)

    assert_allclose(logits_ref, logits)


if __name__ == "__main__":
    test_opt_125M()

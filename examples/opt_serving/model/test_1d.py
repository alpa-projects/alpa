"""Test the correctness of cache implementation."""
import jax
import jax.numpy as jnp
import numpy as np

from alpa.testing import assert_allclose
from opt_serving.model import opt_model, opt_model_1d


def run_2d_model_ref(name, np_weights_folder, input_id_list):
    # Init 2D model
    print("Running 2D OPT model", flush=True)
    config = opt_model.get_opt_config(name, dtype=jnp.float32)
    model_2d, params_2d = opt_model.init_model_aval(config)
    params_2d = opt_model.load_params_np(params_2d, np_weights_folder, config)
    params_2d = jax.tree_map(jnp.array, params_2d)

    # Make cache for each input
    cache_2ds = [
        opt_model.init_cache_np(config, 1) for _ in range(len(input_id_list))
    ]

    # Get expected results
    @jax.jit
    def inference_step_2d(params, batch):
        output = model_2d.apply(params,
                                batch["input_ids"],
                                batch["position_ids"],
                                attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    # Run each prompt individually because their lengths are different
    logits_all = []
    for idx, input_ids in enumerate(input_id_list):
        input_ids = np.array([input_ids], dtype=np.int32)
        position_ids = opt_model.build_position_ids(input_ids, config.pad)

        logits, cache_2ds[idx] = inference_step_2d(
            params_2d, {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "cache": cache_2ds[idx],
            })
        # The logits are in shape (1, seq_len, vocab_size), so we squeeze the batch dimension
        # and concat them together to mimic the output of the 1D model.
        logits_all.append(logits.squeeze())
    logits_all = np.concatenate(logits_all, axis=0)
    print("prompt logits shape:", logits_all.shape, flush=True)
    return logits_all, cache_2ds


def test_opt_125M():
    name = "125M"
    np_weights_folder = f"/home/ubuntu/opt_weights/{name}_np"

    input_id_list = [
        [45942, 2866, 16, 5, 892, 9, 44042, 8],
        [100, 261, 23888, 2426, 16, 10, 21624, 12, 4310, 3034, 9744, 25526, 11],
        [133, 589, 9, 886, 6, 10817, 16, 10, 285],
        [5625, 16, 10, 205, 183, 8, 38, 236, 7],
        [2264, 16, 5, 7440, 9, 16673, 873, 24214, 116],
    ]
    batch_size = len(input_id_list)

    logistic_ref, cache_refs = run_2d_model_ref(name, np_weights_folder,
                                                input_id_list)

    # Init 1D model
    print("Running 1D OPT model", flush=True)
    config = opt_model.get_opt_config(name, dtype=jnp.float32)
    model_1d, params_1d = opt_model_1d.init_model_aval(config, batch_size)
    params_1d = opt_model_1d.load_params_np(params_1d, np_weights_folder,
                                            config)
    params_1d = jax.tree_map(jnp.array, params_1d)
    cache_1ds = [
        opt_model_1d.init_cache_np(config) for _ in range(len(input_id_list))
    ]

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

    # Concate per-input cache together
    cache_1d_flatten = []
    for batched_layer_cache in zip(*cache_1ds):
        batched_layer_key = []
        batched_layer_value = []
        batched_layer_index = []
        for key, value, index in batched_layer_cache:
            batched_layer_key.append(key)
            batched_layer_value.append(value)
            batched_layer_index.append(index)

        key_flatten = np.concatenate(batched_layer_key, axis=0)
        value_flatten = np.concatenate(batched_layer_value, axis=0)
        index_flatten = np.concatenate(batched_layer_index, axis=0)
        cache_1d_flatten.append((key_flatten, value_flatten, index_flatten))

    logits, cache_1d_temp = inference_step_1d(
        params_1d, {
            "input_ids": input_ids_flatten,
            "position_ids": position_ids_flatten,
            "batch_idxs": batch_idxs,
            "cache": cache_1d_flatten,
        })
    print("logits shape:", logits.shape, flush=True)
    assert_allclose(logistic_ref, logits)

    # Recover the cache
    # TODO: Update cache_index
    # Note that assuming the length of 3 input prompts are L1, L2, L3, and the maximum
    # length is L2, then the values of updated cache_index are organized as follows:
    # [<1 x L1>, <0 x (L2-L1)>, <1 x L2>, <1 x L3>, <0 x (L2-L3)>..., 0, ...]
    cache_1ds = [[] for _ in range(batch_size)]
    for key_flatten, value_flatten, index_flatten in cache_1d_temp:
        start = 0
        for batch_idx in range(batch_size):
            end = start + config.max_target_positions
            cache_1ds[batch_idx].append(
                (key_flatten[start:end], value_flatten[start:end],
                 index_flatten[start:end]))
            start = end

    # Compare cache values
    for cache_2d, cache_1d in zip(cache_refs, cache_1ds):
        # Batch
        assert len(cache_2d) == len(cache_1d), \
            "Layer length mismatch: %d vs. %d" % (len(cache_2d), len(cache_1d))

        for layer_2d, layer_1d in zip(cache_2d, cache_1d):
            # Layer
            assert len(layer_2d) == len(layer_1d), \
                "KVI length mismatch: %d vs. %d" % (len(layer_2d), len(layer_1d))

            # Note that cache index formats are not the same, so we skip the comparison.
            key_2d, value_2d, _ = layer_2d
            key_1d, value_1d, _ = layer_1d
            assert_allclose(key_2d.reshape(key_1d.shape), key_1d)
            assert_allclose(value_2d.reshape(value_1d.shape), value_1d)


if __name__ == "__main__":
    test_opt_125M()

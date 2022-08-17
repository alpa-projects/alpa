"""Test the correctness of 1-D OPT implementation."""
import jax
import jax.numpy as jnp
import numpy as np

from alpa.testing import assert_allclose
from opt_serving.model import opt_model, opt_model_1d

# If true, then we run the same batches with 2-D OPT and verify
# the output logtis as well as cache are the same.
VERIFY_RESULT = True

# If true, then the latset logits for both models will be dumped to disk
# as "1d.npy" and "2d.npy" for debugging.
DUMP_LOGITS = False

input_id_list = [
    [45942, 2866, 16, 5, 892, 9, 44042, 8],
    [100, 261, 23888, 2426, 16, 10, 21624, 12, 4310, 3034, 9744, 25526, 11],
    [133, 589, 9, 886, 6, 10817, 16, 10, 285],
    [5625, 16, 10, 205, 183, 8, 38, 236, 7],
    [2264, 16, 5, 7440, 9, 16673, 873, 24214, 116],
]


def init_2d_model(name, np_weights_folder, input_id_list):
    # Init 2D model
    config = opt_model.get_opt_config(name, dtype=jnp.float32)
    model_2d, params_2d = opt_model.init_model_aval(config)
    params_2d = opt_model.load_params_np(params_2d, np_weights_folder, config)
    params_2d = jax.tree_map(jnp.array, params_2d)

    # Make cache for each input
    input_pool_2d = [(input_id, opt_model.init_cache_np(config, 1))
                     for input_id in input_id_list]

    @jax.jit
    def inference_step_2d(params, batch):
        output = model_2d.apply(params,
                                batch["input_ids"],
                                batch["position_ids"],
                                attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    def runner(input_pool):
        # Run each sequence individually because their lengths are different
        is_prompt = [cache[0][2][0] == 0 for _, cache in input_pool]
        logits_all = []
        for idx, (_input_ids, cache) in enumerate(input_pool):
            input_ids = np.array([_input_ids], dtype=np.int32)
            position_ids = opt_model.build_position_ids(input_ids, config.pad)
            if not is_prompt[idx]:
                # Auto-regressive
                input_ids = input_ids[:, -1:]
                position_ids = position_ids[:, -1:]

            logits, updated_cache = inference_step_2d(
                params_2d, {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "cache": cache,
                })
            logits = logits.reshape(logits.shape[1:])
            next_token = np.argmax(logits, axis=-1)[-1]

            # Append the generated token and updated cache.
            #pdb.set_trace()
            input_pool[idx] = (_input_ids + [next_token.tolist()],
                               updated_cache)

            # For debugging
            if DUMP_LOGITS:
                logits_all.append(logits)

        if DUMP_LOGITS:
            logits_all = np.concatenate(logits_all, axis=0)
            jnp.save("2d", logits_all)
        return input_pool

    return runner, input_pool_2d


def init_1d_model(name, np_weights_folder, input_id_list):
    # Init 1D model
    max_batch_size = len(input_id_list)
    config = opt_model.get_opt_config(name, dtype=jnp.float32)
    model_1d, params_1d = opt_model_1d.init_model_aval(config, max_batch_size)
    params_1d = opt_model_1d.load_params_np(params_1d, np_weights_folder,
                                            config)
    params_1d = jax.tree_map(jnp.array, params_1d)

    # Make cache for each input
    input_pool_1d = [(input_id, opt_model_1d.init_cache_np(config))
                     for input_id in input_id_list]

    @jax.jit
    def inference_step_1d(params, batch):
        output = model_1d.apply(params,
                                batch["input_ids"],
                                batch["position_ids"],
                                batch["batch_idxs"],
                                attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    def runner(input_pool):
        batch_size = len(input_pool)
        is_prompt = [cache[0][2][0] == 0 for _, cache in input_pool]

        # Concat inputs together
        input_ids_flatten = []
        position_ids_flatten = []
        batch_idxs = []

        # Promopts must go first, so we use this flag to make sure
        # input pool does not have a pattern like [prompt1, token1, prompt2].
        in_prompt = True

        for idx, (input_ids, cache) in enumerate(input_pool):
            position_ids = opt_model_1d.build_position_ids(
                np.array(input_ids, dtype=np.int32), config.pad).tolist()

            if not is_prompt[idx]:
                # Auto-regressive
                input_ids = input_ids[-1:]
                position_ids = position_ids[-1:]
                in_prompt = False
            else:
                assert in_prompt, "Prompts must be consecutive and before tokens"
            input_ids_flatten += input_ids
            position_ids_flatten += position_ids

            # fused_mmha kernel requires batch_idxs to be in shape sum(input_ids). Each element
            # in batch_idxs is the ID of the corresponding input sequence (starting from 1).
            batch_idxs += np.full(shape=(len(input_ids),),
                                  fill_value=idx + 1,
                                  dtype=np.int32).tolist()

        input_ids_flatten = np.array(input_ids_flatten, dtype=np.int32)
        position_ids_flatten = np.array(position_ids_flatten, dtype=np.int32)
        batch_idxs = np.array(batch_idxs, dtype=np.int32)

        # Concate per-input cache together and generate cache index.
        # Note that assuming the valid cache length of 3 inputs are L1, L2, L3,
        # and the maximum length is L2, then the cache is organized as follows:
        # [<1xL1>, <0x(L2-L1+1)>, <1xL2>, <0x1>, <1xL3>, <0x(L2-L3+1)>, 0, ...]
        # And the total length is BxP, where B is the batch size and P is the
        # maximum cache length. Here we set P to be the max_target_position.
        target_cache_len = max([cache[0][2][0] for _, cache in input_pool]) + 1
        cache_1d_flatten = []
        head_dim = config.decoder_embed_dim // config.decoder_attention_heads
        for batched_layer_cache in zip(*[cache for _, cache in input_pool]):
            batched_layer_key = []
            batched_layer_value = []
            batched_layer_index = []
            for bidx, (key, value, index) in enumerate(batched_layer_cache):
                if is_prompt[bidx]:
                    continue
                curr_valid_len = index[0]
                batched_layer_key.append(key[:target_cache_len])
                batched_layer_value.append(value[:target_cache_len])

                seq_ids = np.full(
                    (curr_valid_len,),  # Valid cache length
                    bidx + 1,  # Sequence ID starting from 1
                    dtype=np.int32)
                batched_layer_index.append(
                    np.concatenate([
                        seq_ids,
                        np.zeros((target_cache_len - curr_valid_len,),
                                 dtype=np.int32)
                    ]))

            if batched_layer_key:
                key_flatten = np.concatenate(batched_layer_key, axis=0)
                value_flatten = np.concatenate(batched_layer_value, axis=0)
                index_flatten = np.concatenate(batched_layer_index, axis=0)
            else:
                # In the case of all prompts we just put empty cache and pad all 0s.
                key_flatten = np.empty(
                    (0, config.decoder_attention_heads, head_dim),
                    dtype=np.float32)
                value_flatten = np.empty(
                    (0, config.decoder_attention_heads, head_dim),
                    dtype=np.float32)
                index_flatten = np.empty((0,), dtype=np.int32)

            # Pad 0s to fixed length: batch_size * max_target_positions
            pad_len = batch_size * config.max_target_positions - index_flatten.shape[
                0]
            key_val_pad = np.zeros((pad_len,) + key_flatten.shape[1:],
                                   dtype=np.float32)
            key_flatten = np.concatenate([key_flatten, key_val_pad], axis=0)
            value_flatten = np.concatenate([value_flatten, key_val_pad], axis=0)

            index_pad = np.zeros((pad_len,), dtype=np.int32)
            index_flatten = np.concatenate([index_flatten, index_pad], axis=0)
            cache_1d_flatten.append((key_flatten, value_flatten, index_flatten))

        logits, cache_1d_updated = inference_step_1d(
            params_1d, {
                "input_ids": input_ids_flatten,
                "position_ids": position_ids_flatten,
                "batch_idxs": batch_idxs,
                "cache": cache_1d_flatten,
            })
        if DUMP_LOGITS:
            jnp.save("1d", logits)

        updated_lengths = [
            len(ids) if prompt else 1
            for prompt, (ids, _) in zip(is_prompt, input_pool)
        ]

        # Recover and update the cache.
        # The output cache only includes the new values of the current generated token,
        # so we perform dynamic slice update on the original cache.
        for lidx, (key_flatten, value_flatten) in enumerate(cache_1d_updated):
            update_start = 0
            key_flatten = key_flatten.squeeze()
            value_flatten = value_flatten.squeeze()
            for bidx in range(batch_size):
                cache = input_pool[bidx][1]
                curr_index = cache[lidx][2][0]

                # Determine the length of updated cache.
                update_end = update_start + updated_lengths[bidx]
                new_index = curr_index + updated_lengths[bidx]

                # Slice update.
                cache[lidx][0][curr_index:new_index] = \
                    key_flatten[update_start:update_end]
                cache[lidx][1][curr_index:new_index] = \
                    value_flatten[update_start:update_end]
                cache[lidx][2][0] = new_index
                input_pool[bidx] = (input_pool[bidx][0], cache)

                update_start = update_end

        # Append the generated token.
        update_start = 0
        update_end = 0
        for bidx, (_input_ids, cache) in enumerate(input_pool):
            update_end = update_start + updated_lengths[bidx]
            next_token = np.argmax(logits[update_start:update_end], axis=-1)[-1]
            input_pool[bidx] = (_input_ids + [next_token.tolist()], cache)
            update_start = update_end
        assert update_end == logits.shape[0]
        return input_pool

    return runner, input_pool_1d


def verify_status(result_1d, result_2d):
    """Verify the current sequence and cache values."""
    print("Verifying results...", flush=True, end="")
    success = True

    for bidx, ((seq_1d, cache_1d),
               (seq_2d, cache_2d)) in enumerate(zip(result_1d, result_2d)):
        # Verify sequence.
        try:
            assert_allclose(seq_1d, seq_2d)
        except AssertionError as err:
            print("Result of seq %d does not match: %s" % (bidx, str(err)))
            success = False

        # Verify cache.
        for lidx, (layer_2d, layer_1d) in enumerate(zip(cache_2d, cache_1d)):
            # Layer
            assert len(layer_2d) == len(layer_1d), \
                "KV length mismatch: %d vs. %d" % (len(layer_2d), len(layer_1d))

            key_1d, value_1d, _ = layer_1d
            key_2d, value_2d, _ = layer_2d
            key_2d = key_2d.reshape(key_1d.shape)
            value_2d = value_2d.reshape(value_1d.shape)
            for cidx, (col_2d, col_1d) in enumerate(zip(key_2d, key_1d)):
                try:
                    assert_allclose(col_2d, col_1d)
                except AssertionError as err:
                    print("KV value mismatch for seq %d at layer %d column %d" %
                          (bidx, lidx, cidx))
                    print(str(err))
                    success = False
    if not success:
        raise RuntimeError("Failed")
    print("passed", flush=True)


def simulate_serving(runner_n_pool, ref_runner_n_pool=None):
    """Simulate model serving."""

    def batch_first_N_prompt(runner, prompt_pool, msg_prefix=""):
        """Batch 1 (prompt * 3)."""
        ret = runner(prompt_pool[2:])
        print("%s Batch 1 (prompt * 3) done" % msg_prefix, flush=True)
        return ret

    def batch_mixed(runner, prompt_pool, decode_pool, msg_prefix=""):
        """Batch 2 (prompt * 2, decoding * 3)."""
        ret = runner(prompt_pool[:2] + decode_pool)
        print("%s Batch 2 (prompt * 2, decoding * 3) done" % msg_prefix,
              flush=True)
        return ret

    def batch_all_decode(runner, decode_pool, msg_prefix="", batch_idx=3):
        """Batch 3+ (decoding * 5)."""
        ret = runner(decode_pool)
        print("%s Batch %d (decoding * 5) done" % (msg_prefix, batch_idx),
              flush=True)
        return ret

    runner, input_pool = runner_n_pool

    verify = False
    ref_runner, ref_input_pool = None, None
    if ref_runner_n_pool is not None:
        verify = True
        ref_runner, ref_input_pool = ref_runner_n_pool

    decode_pool = batch_first_N_prompt(runner, input_pool, "1D")
    if verify:
        ref_decode_pool = batch_first_N_prompt(ref_runner, ref_input_pool, "2D")
        verify_status(decode_pool, ref_decode_pool)

    decode_pool = batch_mixed(runner, input_pool, decode_pool, "1D")
    if verify:
        ref_decode_pool = batch_mixed(ref_runner, ref_input_pool,
                                      ref_decode_pool, "2D")
        verify_status(decode_pool, ref_decode_pool)

    for idx in range(5):
        batch_idx = idx + 3
        decode_pool = batch_all_decode(runner, decode_pool, "1D", batch_idx)
        if verify:
            ref_decode_pool = batch_all_decode(ref_runner, ref_decode_pool,
                                               "2D", batch_idx)
            verify_status(decode_pool, ref_decode_pool)


def test_opt_125M():
    name = "125M"
    np_weights_folder = f"/home/ubuntu/opt_weights/{name}_np"

    model_n_pool_1d = init_1d_model(name, np_weights_folder, input_id_list)

    model_n_pool_2d = None
    if VERIFY_RESULT:
        model_n_pool_2d = init_2d_model(name, np_weights_folder, input_id_list)
    else:
        print("Skipping result verification")

    simulate_serving(model_n_pool_1d, model_n_pool_2d)

    if DUMP_LOGITS:
        print("The latest logits are dumped to 1d.npy "
              "(and 2d.npy if VERIFY_RESULT=True)")


if __name__ == "__main__":
    test_opt_125M()

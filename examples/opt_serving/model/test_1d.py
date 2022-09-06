"""Test the correctness of 1-D OPT implementation."""
import time
from typing import List, Dict

import copy

import jax
import jax.numpy as jnp
import numpy as np
import os
from functools import partial

from alpa.testing import assert_allclose
from alpa.util import print_used_time
from opt_serving.model import opt_model, opt_model_1d
from alpa.collective.worker_nccl_util_cupy import jax_tensor_to_cupy, cupy_to_jax_tensor

# If true, then we run the same batches with 2-D OPT and verify
# the output logits as well as cache are the same.
VERIFY_RESULT = True

# If true, then the latest logits for both models will be dumped to disk
# as "1d.npy" and "2d.npy" for debugging.
DUMP_LOGITS = False

# total input len -- #tokens in the batch
N = 64
# total cache length
M = 512
MAX_CACHE_LEN_PER_SEQ = 32


input_id_list = [
    [45942, 2866, 16, 5, 892, 9, 44042, 8],
    [100, 261, 23888, 2426, 16, 10, 21624, 12, 4310, 3034, 9744, 25526, 11],
    [133, 589, 9, 886, 6, 10817, 16, 10, 285],
    [5625, 16, 10, 205, 183, 8, 38, 236, 7],
    [2264, 16, 5, 7440, 9, 16673, 873, 24214, 116],
    # Second batch
    [32826, 16, 5, 812, 343, 9],
    [2264, 109, 47, 206, 59, 5, 499, 9, 28850, 1975, 37079, 116],
    [2264, 109, 47, 206, 59, 5, 3099, 9, 301, 116],
    [19195, 140, 16, 5, 394, 9],
    [534, 10311, 12, 246, 16, 10, 739, 2777, 1421, 14, 16, 4453, 9],
]


tic = 0.0
def record_time(start=False):
    global tic
    if start:
        tic = time.time()
        return
    t = time.time() - tic
    tic = time.time()
    return t


class Jax1DInput:
    def __init__(self, input_tokens=None, input_sentence_ids=None, kv_caches=None, kv_cache_ids=None, num_prev_tokens=None):
        self.input_tokens = input_tokens
        self.input_sentence_ids = input_sentence_ids
        self.kv_caches = kv_caches
        self.kv_cache_ids = kv_cache_ids
        self.num_prev_tokens = num_prev_tokens

    def __getitem__(self, indices):
        return Jax1DInput(
            self.input_tokens[indices],
            self.input_sentence_ids[indices],
            self.kv_caches,
            self.kv_cache_ids,
            self.num_prev_tokens)


def init_2d_inference_step(name, np_weights_folder):
    # Init 2D model
    config = opt_model.get_opt_config(name, dtype=jnp.float32)
    model_2d, params_2d = opt_model.init_model_aval(config)
    params_2d = opt_model.load_params_np(params_2d, np_weights_folder, config)
    params_2d = jax.tree_map(jnp.array, params_2d)
    cache_2d = opt_model.init_cache_np(config, 1)

    @jax.jit
    def inference_step_2d(params, batch):
        output = model_2d.apply(params,
                                batch["input_ids"],
                                batch["position_ids"],
                                attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    return inference_step_2d, params_2d, cache_2d, config


def runner_2d(model, input_pool):
    # Run each sequence individually because their lengths are different
    inference_step, params, config = model
    is_prompt = [cache[0][2][0] == 0 for _, cache in input_pool]
    logits_all = []
    execution_cost = {}

    record_time(start=True)
    for idx, (_input_ids, cache) in enumerate(input_pool):
        input_ids = np.array([_input_ids], dtype=np.int32)
        position_ids = opt_model.build_position_ids(input_ids, config.pad)
        if not is_prompt[idx]:
            # Auto-regressive
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[:, -1:]
        logits, updated_cache = inference_step(
            params, {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "cache": cache,
            })
        logits = logits.reshape(logits.shape[1:])
        next_token = np.argmax(logits, axis=-1)[-1]
        # Append the generated token and updated cache.
        input_pool[idx] = (_input_ids + [next_token.tolist()],
                           updated_cache)

        # For debugging
        if DUMP_LOGITS:
            logits_all.append(logits)
    execution_cost["compute"] = record_time()

    if DUMP_LOGITS:
        logits_all = np.concatenate(logits_all, axis=0)
        jnp.save("2d", logits_all)
    return input_pool, execution_cost


def setup(mode,
          input_id_list,
          model="jax/opt-125m",
          np_weights_folder="~/opt_weights/"):
    name = model.split("-")[1].upper()
    path = os.path.join(np_weights_folder + f"{name}_np")
    if mode == "2d":
        inference_step, params, cache, config = init_2d_inference_step(name, path)
        model_tuple = tuple([inference_step, params, config])
        input_pool = [(input_id, copy.deepcopy(cache)) for input_id in input_id_list]
    else:
        assert mode == "1d", "only support `1d` or `2d`."
        inference_step, params, kv_caches, kv_caches_cupy, config = init_1d_inference_step(name, path)
        model_tuple = tuple([inference_step, params, config])
        # make input pool for 1D
        # input_tokens = input_id_list
        # input_sentence_ids = np.arange(1, len(input_tokens) + 1)
        # kv_cache_ids = np.zeros((M, ), dtype=np.int32)
        # num_prev_tokens: Dict[int, int] = {}
        # input_pool = tuple([input_tokens, input_sentence_ids, kv_caches, kv_caches_cupy, kv_cache_ids, num_prev_tokens])
        input_pool = Jax1DInput(input_id_list,
                                [i for i in range(1, len(input_id_list) + 1)],
                                kv_caches,
                                np.zeros((M, ), dtype=np.int32),
                                {})

    return model_tuple, input_pool


def init_1d_inference_step(name, np_weights_folder, total_input_len=N, total_cache_len=M):
    config = opt_model.get_opt_config(name, dtype=jnp.float32)
    model_1d, params_1d = opt_model_1d.init_model_aval_v2(config, total_input_len, total_cache_len)
    params_1d = opt_model_1d.load_params_np(params_1d, np_weights_folder,
                                            config)
    params_1d = jax.tree_map(jnp.array, params_1d)
    # Init caches (here we allocate memory)
    kv_caches = opt_model_1d.init_cache_np_v2(config, total_cache_len)
    kv_caches = [(jnp.asarray(k), jnp.asarray(v)) for k, v in kv_caches]
    kv_caches_cupy = [(jax_tensor_to_cupy(k), jax_tensor_to_cupy(v)) for k, v in kv_caches]

    @jax.jit
    def inference_step_1d(params, batch):
        output = model_1d.apply(params,
                                batch["input_ids"],
                                batch["position_ids"],
                                attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    return inference_step_1d, params_1d, kv_caches, kv_caches_cupy, config


def runner_1d(model, input_pool):
    # input_tokens should not include paddings.

    inference_step, params, config = model
    input_tokens = input_pool.input_tokens
    input_sentence_ids = input_pool.input_sentence_ids
    kv_caches = input_pool.kv_caches
    kv_cache_ids = input_pool.kv_cache_ids
    num_prev_tokens = input_pool.num_prev_tokens

    kv_caches_cupy = [(jax_tensor_to_cupy(k), jax_tensor_to_cupy(v)) for k, v in kv_caches]
    # input_tokens, input_sentence_ids, kv_caches, kv_caches_cupy, kv_cache_ids, num_prev_tokens = input_pool

    for input_sentence in input_tokens:
        assert config.pad not in input_sentence

    # TODO(Woosuk): Reorder the input sentences.

    # Pre-update num_prev_tokens.
    for i, sentence_id in enumerate(input_sentence_ids):
        if sentence_id in num_prev_tokens:
            assert len(input_tokens[i]) == 1
            assert num_prev_tokens[sentence_id] > 0
        else:
            num_prev_tokens[sentence_id] = 0

    # Generate inputs.
    input_1d = sum(input_tokens, [])
    input_1d = input_1d + [config.pad] * (N - len(input_1d))
    input_1d = jnp.asarray(input_1d, dtype=jnp.int32)

    # Generate sentence ids.
    assert len(input_tokens) == len(input_sentence_ids)
    input_index_1d = []
    for i, sentence_id in enumerate(input_sentence_ids):
        input_index_1d.extend([sentence_id] * len(input_tokens[i]))
    input_index_1d = input_index_1d + [0] * (N - len(input_index_1d))
    input_index_1d = np.array(input_index_1d, dtype=np.int32)

    # Generate position ids.
    position_id_1d = []
    for i, sentence_id in enumerate(input_sentence_ids):
        start_idx = 1 + config.pad + num_prev_tokens[sentence_id]
        position_ids = list(range(start_idx, start_idx + len(input_tokens[i])))
        position_id_1d.extend(position_ids)
    position_id_1d = position_id_1d + [config.pad] * (N - len(position_id_1d))
    position_id_1d = jnp.asarray(position_id_1d, dtype=jnp.int32)

    assert MAX_CACHE_LEN_PER_SEQ >= max(num_prev_tokens.values())
    assert MAX_CACHE_LEN_PER_SEQ * len(num_prev_tokens) <= kv_cache_ids.shape[0]
    os.environ['FT_INPUT_INDEX_ADDR'] = str(input_index_1d.ctypes.data)
    os.environ['FT_CACHE_INDEX_ADDR'] = str(kv_cache_ids.ctypes.data)
    os.environ['FT_MAX_CACHE_LEN_PER_SEQ'] = str(MAX_CACHE_LEN_PER_SEQ)

    batch = {
        "input_ids": input_1d,
        "position_ids": position_id_1d,
        "cache": kv_caches
    }

    logits, kv = inference_step(params, batch)

    # Get the output tokens.
    logits = np.array(logits)
    outputs = []
    output_idx = -1
    for i, sentence_id in enumerate(input_sentence_ids):
        output_idx += len(input_tokens[i])
        outputs.append([int(np.argmax(logits[output_idx]))])

    # Update kv_caches.
    for layer_idx, (key_1d, value_1d) in enumerate(kv):
        k_cache, v_cache = kv_caches_cupy[layer_idx]
        key_1d = jax_tensor_to_cupy(key_1d)
        value_1d = jax_tensor_to_cupy(value_1d)
        idx = 0
        for i, sentence_id in enumerate(input_sentence_ids):
            # FIXME
            cache_idx = (sentence_id - 1) * MAX_CACHE_LEN_PER_SEQ + num_prev_tokens[sentence_id]
            for _ in range(len(input_tokens[i])):
                k_cache[cache_idx] = key_1d[idx]
                v_cache[cache_idx] = value_1d[idx]
                kv_cache_ids[cache_idx] = sentence_id
                cache_idx += 1
                idx += 1

    # Post-update num_prev_tokens.
    for i, sentence_id in enumerate(input_sentence_ids):
        # TODO: Handle EOS here.
        num_prev_tokens[sentence_id] += len(input_tokens[i])

    kv_caches = [(cupy_to_jax_tensor(k), cupy_to_jax_tensor(v)) for k, v in kv_caches_cupy]
    # TODO(Hao): make outputs as an input_pool
    output_pool = Jax1DInput(outputs, input_sentence_ids, kv_caches, kv_cache_ids, num_prev_tokens)
    return output_pool

# def runner_1d(input_pool, inference_step, params, config):
#     batch_size = len(input_pool)
#     is_prompt = [cache[0][2][0] == 0 for _, cache in input_pool]
#     execution_cost = {}
#
#     record_time(start=True)
#     # Concat inputs together
#     input_ids_flatten = []
#     position_ids_flatten = []
#     batch_idxs = []
#
#     # Prompts must go first, so we use this flag to make sure
#     # input pool does not have a pattern like [prompt1, token1, prompt2].
#     in_prompt = True
#     for idx, (input_ids, cache) in enumerate(input_pool):
#         position_ids = opt_model_1d.build_position_ids(
#             np.array(input_ids, dtype=np.int32), config.pad).tolist()
#
#         if not is_prompt[idx]:
#             # Auto-regressive
#             input_ids = input_ids[-1:]
#             position_ids = position_ids[-1:]
#             in_prompt = False
#         else:
#             assert in_prompt, "Prompts must be consecutive and before tokens"
#         input_ids_flatten += input_ids
#         position_ids_flatten += position_ids
#
#         # fused_mmha kernel requires batch_idxs to be in shape sum(input_ids). Each element
#         # in batch_idxs is the ID of the corresponding input sequence (starting from 1).
#         batch_idxs += np.full(shape=(len(input_ids),),
#                               fill_value=idx + 1,
#                               dtype=np.int32).tolist()
#
#     input_ids_flatten = np.array(input_ids_flatten, dtype=np.int32)
#     position_ids_flatten = np.array(position_ids_flatten, dtype=np.int32)
#     batch_idxs = np.array(batch_idxs, dtype=np.int32)
#     execution_cost["flattening"] = record_time()
#
#     # Concate per-input cache together and generate cache index.
#     # Note that assuming the valid cache length of 3 inputs are L1, L2, L3,
#     # and the maximum length is L2, then the cache is organized as follows:
#     # [<1xL1>, <0x(L2-L1+1)>, <1xL2>, <0x1>, <1xL3>, <0x(L2-L3+1)>, 0, ...]
#     # And the total length is BxP, where B is the batch size and P is the
#     # maximum cache length. Here we set P to be the max_target_position.
#     target_cache_len = max([cache[0][2][0] for _, cache in input_pool]) + 1
#     cache_1d_flatten = []
#     head_dim = config.decoder_embed_dim // config.decoder_attention_heads
#     for batched_layer_cache in zip(*[cache for _, cache in input_pool]):
#         batched_layer_key = []
#         batched_layer_value = []
#         batched_layer_index = []
#         for bidx, (key, value, index) in enumerate(batched_layer_cache):
#             if is_prompt[bidx]:
#                 continue
#             curr_valid_len = index[0]
#             batched_layer_key.append(key[:target_cache_len])
#             batched_layer_value.append(value[:target_cache_len])
#
#             seq_ids = np.full(
#                 (curr_valid_len,),  # Valid cache length
#                 bidx + 1,  # Sequence ID starting from 1
#                 dtype=np.int32)
#             batched_layer_index.append(
#                 np.concatenate([
#                     seq_ids,
#                     np.zeros((target_cache_len - curr_valid_len,),
#                              dtype=np.int32)
#                 ]))
#
#         if batched_layer_key:
#             key_flatten = np.concatenate(batched_layer_key, axis=0)
#             value_flatten = np.concatenate(batched_layer_value, axis=0)
#             index_flatten = np.concatenate(batched_layer_index, axis=0)
#         else:
#             # In the case of all prompts we just put empty cache and pad all 0s.
#             key_flatten = np.empty(
#                 (0, config.decoder_attention_heads, head_dim),
#                 dtype=np.float32)
#             value_flatten = np.empty(
#                 (0, config.decoder_attention_heads, head_dim),
#                 dtype=np.float32)
#             index_flatten = np.empty((0,), dtype=np.int32)
#
#         # Pad 0s to fixed length: batch_size * max_target_positions
#         pad_len = batch_size * config.max_target_positions - index_flatten.shape[
#             0]
#         key_val_pad = np.zeros((pad_len,) + key_flatten.shape[1:],
#                                dtype=np.float32)
#         key_flatten = np.concatenate([key_flatten, key_val_pad], axis=0)
#         value_flatten = np.concatenate([value_flatten, key_val_pad], axis=0)
#
#         index_pad = np.zeros((pad_len,), dtype=np.int32)
#         index_flatten = np.concatenate([index_flatten, index_pad], axis=0)
#         cache_1d_flatten.append((key_flatten, value_flatten, index_flatten))
#
#     execution_cost["cache"] = record_time()
#     # FIXME(Hao): Set env var for this workaround:
#     #  https://github.com/WoosukKwon/1d-mmha-op/blob/main/lib/kernels.cc.cu#L53
#     host_device = jax.devices('cpu')[0]
#     host_input_index = jax.device_put(batch_idxs, device=host_device)
#     host_cache_index = jax.device_put(cache_1d_flatten[0][2], device=host_device)
#     os.environ["FT_INPUT_INDEX_ADDR"] = str(host_input_index.unsafe_buffer_pointer())
#     os.environ["FT_CACHE_INDEX_ADDR"] = str(host_cache_index.unsafe_buffer_pointer())
#     execution_cost["set_env"] = record_time()
#
#     logits, cache_1d_updated = inference_step(
#         params, {
#             "input_ids": input_ids_flatten,
#             "position_ids": position_ids_flatten,
#             "batch_idxs": batch_idxs,
#             "cache": cache_1d_flatten,
#         })
#     execution_cost["compute"] = record_time()
#     if DUMP_LOGITS:
#         jnp.save("1d", logits)
#
#     updated_lengths = [
#         len(ids) if prompt else 1
#         for prompt, (ids, _) in zip(is_prompt, input_pool)
#     ]
#
#     # Recover and update the cache.
#     # The output cache only includes the new values of the current generated token,
#     # so we perform dynamic slice update on the original cache.
#     for lidx, (key_flatten, value_flatten) in enumerate(cache_1d_updated):
#         update_start = 0
#         key_flatten = key_flatten.squeeze()
#         value_flatten = value_flatten.squeeze()
#         for bidx in range(batch_size):
#             cache = input_pool[bidx][1]
#             curr_index = cache[lidx][2][0]
#
#             # Determine the length of updated cache.
#             update_end = update_start + updated_lengths[bidx]
#             new_index = curr_index + updated_lengths[bidx]
#
#             # Slice update.
#             cache[lidx][0][curr_index:new_index] = \
#                 key_flatten[update_start:update_end]
#             cache[lidx][1][curr_index:new_index] = \
#                 value_flatten[update_start:update_end]
#             cache[lidx][2][0] = new_index
#             input_pool[bidx] = (input_pool[bidx][0], cache)
#
#             update_start = update_end
#
#     # Append the generated token.
#     update_start = 0
#     update_end = 0
#     for bidx, (_input_ids, cache) in enumerate(input_pool):
#         update_end = update_start + updated_lengths[bidx]
#         next_token = np.argmax(logits[update_start:update_end], axis=-1)[-1]
#         input_pool[bidx] = (_input_ids + [next_token.tolist()], cache)
#         update_start = update_end
#     assert update_end == logits.shape[0]
#     execution_cost["reorder"] = record_time()
#
#     return input_pool, execution_cost

# def init_1d_runner(name, np_weights_folder, input_id_list):
#     inference_step, params, cache, config = init_1d_inference_step(name, np_weights_folder)
#
#     # Make cache for each input
#     input_pool_1d = [(input_id, copy.deepcopy(cache))
#                      for input_id in input_id_list]
#
#     return runner, input_pool_1d


def verify_status(result_1d, result_2d):
    """Verify the current sequence and cache values."""
    print("Verifying results...", flush=True, end="")
    success = True

    tokens_1d = result_1d.input_tokens
    caches_1d = result_1d.kv_caches

    new_results_1d = [(tokens, caches_1d[i]) for i, tokens in enumerate(tokens_1d)]

    for bidx, ((seq_1d, cache_1d),
               (seq_2d, cache_2d)) in enumerate(zip(new_results_1d, result_2d)):
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

def verify_next_token(output_pool_1d, output_pool_2d):
    print("Verifying next token prediction...", flush=True, end="")
    success = True
    tokens_1d = [token_list[0] for token_list in output_pool_1d.input_tokens]
    tokens_2d = [sentence[0][-1] for sentence in output_pool_2d]

    for sentence_id, (next_token_1d, next_token_2d) in enumerate(zip(tokens_1d, tokens_2d)):
        try:
            assert_allclose(next_token_1d, next_token_2d)
        except AssertionError as err:
            print("Result of seq %d does not match: %s" % (sentence_id, str(err)))
            success = False
    if not success:
        raise RuntimeError("Failed")
    print("passed", flush=True)

def verify_caches(output_pool_1d, output_pool_2d):
    print("Verifying caches...", flush=True, end="")
    success = True

    # Process cache 2D
    caches_2d = [list(sentence[1]) for sentence in output_pool_2d]
    kv_caches = output_pool_1d.kv_caches
    sentence_ids = output_pool_1d.input_sentence_ids

    # re-organize cache:
    new_caches_2d = [None for _ in range(len(sentence_ids))]
    for i, id in enumerate(sentence_ids):
        new_caches_2d[id - 1] = caches_2d[i]
    caches_2d = new_caches_2d

    caches_1d = []
    for i in range(len(sentence_ids)):
        caches_1d.append([])
        for j in range(len(kv_caches)):
            caches_1d[i].append(None)
    for layer_idx, (key_1d, value_1d) in enumerate(kv_caches):
        for i, sentence_id in enumerate(sentence_ids):
            start = (sentence_id - 1) * MAX_CACHE_LEN_PER_SEQ
            end = sentence_id * MAX_CACHE_LEN_PER_SEQ
            if caches_1d[sentence_id - 1][layer_idx] is None:
                caches_1d[sentence_id - 1][layer_idx] = (
                    np.zeros_like(caches_2d[sentence_id-1][layer_idx][0]),
                    np.zeros_like(caches_2d[sentence_id-1][layer_idx][1]))
            caches_1d[sentence_id - 1][layer_idx][0][0, 0:MAX_CACHE_LEN_PER_SEQ,:,:] = key_1d[start:end]
            caches_1d[sentence_id - 1][layer_idx][1][0, 0:MAX_CACHE_LEN_PER_SEQ,:,:] = value_1d[start:end]

    for bidx, (cache_1d, cache_2d) in enumerate(zip(caches_1d, caches_2d)):
        # Verify cache.
        for lidx, (layer_2d, layer_1d) in enumerate(zip(cache_2d, cache_1d)):
            # Layer
            layer_2d = layer_2d[:2]
            assert len(layer_2d) == len(layer_1d), \
                "KV length mismatch: %d vs. %d" % (len(layer_2d), len(layer_1d))

            key_1d, value_1d = layer_1d
            key_2d, value_2d = layer_2d
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

    decode_pool, _ = batch_first_N_prompt(runner, input_pool, "1D")
    if verify:
        ref_decode_pool, _ = batch_first_N_prompt(ref_runner, ref_input_pool, "2D")
        verify_status(decode_pool, ref_decode_pool)

    decode_pool, _ = batch_mixed(runner, input_pool, decode_pool, "1D")
    if verify:
        ref_decode_pool, _ = batch_mixed(ref_runner, ref_input_pool,
                                      ref_decode_pool, "2D")
        verify_status(decode_pool, ref_decode_pool)

    for idx in range(5):
        batch_idx = idx + 3
        decode_pool, _ = batch_all_decode(runner, decode_pool, "1D", batch_idx)
        if verify:
            ref_decode_pool, _ = batch_all_decode(ref_runner, ref_decode_pool,
                                               "2D", batch_idx)
            verify_status(decode_pool, ref_decode_pool)



def test_opt_125M():
    name = "jax/opt-125m"
    np_weights_folder = f"/home/ubuntu/opt_weights/"

    batch_size = 5

    # init
    model_1d, input_pool_1d = setup("1d", input_id_list, name, np_weights_folder)
    model_2d, input_pool_2d = setup("2d", input_id_list, name, np_weights_folder)

    # batch the first 5 prompts
    # input_tokens, input_sentence_ids, kv_caches, kv_caches_cupy, kv_cache_ids, num_prev_tokens
    output_pool_1d = runner_1d(model_1d, input_pool_1d[:batch_size])
    output_pool_2d, _ = runner_2d(model_2d, input_pool_2d[:batch_size])
    verify_next_token(output_pool_1d, output_pool_2d)
    verify_caches(output_pool_1d, output_pool_2d)

    # batch the second 5 prompts and the first 5 words
    output_pool_1d = Jax1DInput(
        input_id_list[batch_size:] + output_pool_1d.input_tokens,
        [i+1 for i in range(batch_size, len(input_id_list))] + output_pool_1d.input_sentence_ids,
        output_pool_1d.kv_caches,
        output_pool_1d.kv_cache_ids,
        output_pool_1d.num_prev_tokens
    )
    output_pool_2d = input_pool_2d[batch_size:] + output_pool_2d
    output_pool_2d, _ = runner_2d(model_2d, output_pool_2d)
    output_pool_1d = runner_1d(model_1d, output_pool_1d)

    verify_next_token(output_pool_1d, output_pool_2d)
    verify_caches(output_pool_1d, output_pool_2d)


    exit(1)
    # TODO(Hao): do something with the pool
    # batch all decode
    output_pool_1d = runner_1d(output_pool_1d)
    output_pool_2d = runner_2d(output_pool_2d)
    verify_status(output_pool_1d, output_pool_2d)


if __name__ == "__main__":
    test_opt_125M()

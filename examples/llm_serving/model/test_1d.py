"""Test the correctness of 1-D OPT implementation."""
import time
from typing import List, Dict

import copy

import jax
import jax.numpy as jnp
import numpy as np
import cupy
import os

from alpa.testing import assert_allclose
from llm_serving.model import opt_model, opt_model_1d
from alpa.collective.worker_nccl_util_cupy import jax_tensor_to_cupy
from alpa.timer import timers


# If true, then the latest logits for both models will be dumped to disk
# as "1d.npy" and "2d.npy" for debugging.
DUMP_LOGITS = False

# total input len -- #tokens in the batch
N = 64
# total cache length
M = 512 * 2
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


class Jax1DInput:
    """Temp class for benchmarking and runner."""
    def __init__(self,
                 input_tokens,
                 input_sentence_ids,
                 kv_caches,
                 kv_cache_ids,
                 num_prev_tokens):
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


def pad_batch(input_batch: List[List[int]], pad=1):
    max_len = max(len(sen) for sen in input_batch)
    input_batch_padded = copy.deepcopy(input_batch)
    for sen in input_batch_padded:
        if len(sen) < max_len:
            for _ in range(max_len - len(sen)):
                sen.append(pad)
    return input_batch_padded


# TODO(Hao): integrate this with the get_model in our OPT serving
def setup(mode: str,
          input_id_list: List[List[int]],
          model="jax/opt-125m",
          np_weights_folder="~/opt_weights/",
          batch_size=1,
          cache_size=M,
          max_cache_len_per_seq=MAX_CACHE_LEN_PER_SEQ):
    name = model.split("-")[1].upper()
    path = os.path.join(np_weights_folder + f"{name}_np")
    if mode == "2d":
        model_tuple = init_2d_inference_step(name, path, batch_size=batch_size)
        config = model_tuple[-1]
        cache = opt_model.init_cache_np(config, batch_size)
        assert len(input_id_list) % batch_size == 0, "Do not support padding batch now..."
        num_batch = len(input_id_list) // batch_size
        input_pool = []
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size
            batch = input_id_list[start:end]
            # pad the batch
            input_pool.append((batch, copy.deepcopy(cache)))
    else:
        assert mode == "1d", "only support `1d` or `2d`."
        model_tuple = init_1d_inference_step(name,
                                             path,
                                             total_input_len=batch_size,
                                             total_cache_len=cache_size)
        global N
        N = batch_size
        global M
        M = cache_size
        global MAX_CACHE_LEN_PER_SEQ
        MAX_CACHE_LEN_PER_SEQ = max_cache_len_per_seq
        config = model_tuple[-1]
        # Init caches (here we allocate memory)
        kv_caches = opt_model_1d.init_cache_np(config, cache_size)
        kv_caches = [(jnp.asarray(k), jnp.asarray(v)) for k, v in kv_caches]
        # kv_caches_cupy = [(jax_tensor_to_cupy(k), jax_tensor_to_cupy(v)) for k, v in kv_caches]

        input_pool = Jax1DInput(input_id_list,
                                [i for i in range(1, len(input_id_list) + 1)],
                                kv_caches,
                                np.zeros((cache_size, ), dtype=np.int32),
                                {})
    return model_tuple, input_pool


def init_2d_inference_step(name, np_weights_folder, batch_size=1):
    # Init 2D model
    config = opt_model.get_opt_config(name, dtype=jnp.float32)
    model_2d, params_2d = opt_model.init_model_aval(config)
    params_2d = opt_model.load_params_np(params_2d, np_weights_folder, config)
    params_2d = jax.tree_map(jnp.array, params_2d)

    @jax.jit
    def inference_step_2d(params, batch):
        output = model_2d.apply(params,
                                batch["input_ids"],
                                batch["position_ids"],
                                attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    return inference_step_2d, params_2d, config


def sync(device_id=0):
    jax.devices()[device_id].synchronize_all_activity()


def runner_2d(model, input_pool):
    inference_step, params, config = model
    is_prompt = [cache[0][2][0] == 0 for _, cache in input_pool]
    logits_all = []
    output_pool = []

    timers("2d_overall").start(sync)
    for idx, (_input_ids, cache) in enumerate(input_pool):
        output_ids = []
        original_len = [len(sen) for sen in _input_ids]
        _input_ids_padded = pad_batch(_input_ids)
        input_ids = np.array(_input_ids_padded, dtype=np.int32)
        position_ids = opt_model.build_position_ids(input_ids, config.pad)
        if not is_prompt[idx]:
            # Auto-regressive
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        timers("2d_compute").start(sync)
        logits, updated_cache = inference_step(
            params, {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "cache": cache,
        })
        timers("2d_compute").suspend(sync)

        next_token = jnp.argmax(logits, axis=-1)
        # Append the generated token and updated cache.
        for i, sen in enumerate(_input_ids):
            output_ids.append(sen + [int(next_token[i][original_len[i]-1])])
        output_pool.append((output_ids, updated_cache))

        # For debugging
        if DUMP_LOGITS:
            logits_all.append(logits)

    timers("2d_compute").stop(sync)
    timers("2d_overall").stop(sync)

    if DUMP_LOGITS:
        logits_all = np.concatenate(logits_all, axis=0)
        jnp.save("2d", logits_all)
    return output_pool


def init_1d_inference_step(name,
                           np_weights_folder,
                           total_input_len=N,
                           total_cache_len=M):
    config = opt_model.get_opt_config(name, dtype=jnp.float32)
    model_1d, params_1d = opt_model_1d.init_model_aval(config, total_input_len, total_cache_len)
    params_1d = opt_model_1d.load_params_np(params_1d, np_weights_folder, config)
    params_1d = jax.tree_map(jnp.array, params_1d)

    @jax.jit
    def inference_step_1d(params, batch):
        output = model_1d.apply(params,
                                batch["input_ids"],
                                batch["position_ids"],
                                attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    return inference_step_1d, params_1d, config


def runner_1d(model, input_pool):
    inference_step, params, config = model
    input_tokens = input_pool.input_tokens
    input_sentence_ids = input_pool.input_sentence_ids
    kv_caches = input_pool.kv_caches
    kv_cache_ids = input_pool.kv_cache_ids
    num_prev_tokens = input_pool.num_prev_tokens

    if isinstance(kv_caches[0][0], np.ndarray):
        kv_caches = [(jnp.asarray(k), jnp.asarray(v)) for k, v in kv_caches]

    kv_caches_cupy = [(jax_tensor_to_cupy(k), jax_tensor_to_cupy(v)) for k, v in kv_caches]
    # input_tokens should not include paddings.
    for input_sentence in input_tokens:
        assert config.pad not in input_sentence

    # TODO(Woosuk): Reorder the input sentences.
    timers("1d_overall").start(sync)
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
    assert N >= len(input_index_1d)
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
    assert MAX_CACHE_LEN_PER_SEQ * len(num_prev_tokens) <= kv_cache_ids.shape[0], \
        f"LHS: {MAX_CACHE_LEN_PER_SEQ * len(num_prev_tokens)}, RHS: {kv_cache_ids.shape[0]}"
    os.environ['FT_INPUT_INDEX_ADDR'] = str(input_index_1d.ctypes.data)
    os.environ['FT_CACHE_INDEX_ADDR'] = str(kv_cache_ids.ctypes.data)
    os.environ['FT_MAX_CACHE_LEN_PER_SEQ'] = str(MAX_CACHE_LEN_PER_SEQ)

    batch = {
        "input_ids": input_1d,
        "position_ids": position_id_1d,
        "cache": kv_caches
    }
    timers("1d_compute").start(sync)
    logits, kv = inference_step(params, batch)
    timers("1d_compute").stop(sync)

    # Get the output tokens.
    next_token = jnp.argmax(logits, axis=-1)
    outputs = []
    output_idx = -1
    next_token = np.array(next_token)
    for i, sentence_id in enumerate(input_sentence_ids):
        output_idx += len(input_tokens[i])
        outputs.append([int(next_token[output_idx])])

    # Update kv_caches.
    timers("1d_update_cache").start(sync)
    for layer_idx, (key_1d, value_1d) in enumerate(kv):

        k_cache, v_cache = kv_caches_cupy[layer_idx]
        key_1d = jax_tensor_to_cupy(key_1d)
        value_1d = jax_tensor_to_cupy(value_1d)
        idx = 0
        for i, sentence_id in enumerate(input_sentence_ids):
            # FIXME
            cache_idx = (sentence_id - 1) * MAX_CACHE_LEN_PER_SEQ + num_prev_tokens[sentence_id]
            k_cache[cache_idx:cache_idx+len(input_tokens[i]),:] = cupy.squeeze(key_1d[idx:idx+len(input_tokens[i]),:])
            v_cache[cache_idx:cache_idx+len(input_tokens[i]),:] = cupy.squeeze(value_1d[idx:idx+len(input_tokens[i]),:])
            kv_cache_ids[cache_idx:cache_idx+len(input_tokens[i])] = sentence_id
            idx += len(input_tokens[i])
    # Post-update num_prev_tokens.
    for i, sentence_id in enumerate(input_sentence_ids):
        # TODO: Handle EOS here.
        num_prev_tokens[sentence_id] += len(input_tokens[i])
    timers("1d_update_cache").stop(sync)

    # kv_caches = [(cupy_to_jax_tensor(k), cupy_to_jax_tensor(v)) for k, v in kv_caches_cupy]
    output_pool = Jax1DInput(outputs, input_sentence_ids, kv_caches, kv_cache_ids, num_prev_tokens)
    timers("1d_overall").stop(sync)
    return output_pool


def verify_next_token(output_pool_1d, output_pool_2d):
    print("Verifying next token prediction...", flush=True, end="")
    success = True
    tokens_1d = [token_list[0] for token_list in output_pool_1d.input_tokens]
    tokens_2d = [sentence[0][0][-1] for sentence in output_pool_2d]

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


def test_opt_125M():
    name = "jax/opt-125m"
    np_weights_folder = f"/home/ubuntu/opt_weights/"
    batch_size = 5

    # init
    batch_size_1d = sum([len(seq) for seq in input_id_list])
    max_seq_len = max([len(seq) for seq in input_id_list])
    cache_size_1d = (max_seq_len + 2) * len(input_id_list)
    model_1d, input_pool_1d = setup("1d", input_id_list, name, np_weights_folder,
                                    batch_size=batch_size_1d,
                                    cache_size=cache_size_1d,
                                    max_cache_len_per_seq=cache_size_1d // len(input_id_list))
    model_2d, input_pool_2d = setup("2d", input_id_list, name, np_weights_folder,
                                    batch_size=1)

    # batch the first 5 prompts
    # input_tokens, input_sentence_ids, kv_caches, kv_caches_cupy, kv_cache_ids, num_prev_tokens
    output_pool_1d = runner_1d(model_1d, input_pool_1d[:batch_size])
    output_pool_2d = runner_2d(model_2d, input_pool_2d[:batch_size])
    verify_next_token(output_pool_1d, output_pool_2d)
    verify_caches(output_pool_1d, output_pool_2d)

    # batch the second 5 prompts and the first 5 words
    # Note(Hao): prompts must go first.
    output_pool_1d = Jax1DInput(
        input_id_list[batch_size:] + output_pool_1d.input_tokens,
        [i+1 for i in range(batch_size, len(input_id_list))] + output_pool_1d.input_sentence_ids,
        output_pool_1d.kv_caches,
        output_pool_1d.kv_cache_ids,
        output_pool_1d.num_prev_tokens
    )
    output_pool_2d = input_pool_2d[batch_size:] + output_pool_2d
    output_pool_2d = runner_2d(model_2d, output_pool_2d)
    output_pool_1d = runner_1d(model_1d, output_pool_1d)
    verify_next_token(output_pool_1d, output_pool_2d)
    verify_caches(output_pool_1d, output_pool_2d)


if __name__ == "__main__":
    test_opt_125M()

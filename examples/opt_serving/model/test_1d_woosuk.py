import os
from typing import Dict, List

import numpy as np
import cupy
import jax
from jax._src.lib import xla_client as xc
import jax.numpy as jnp
from opt_serving.model import opt_model
from opt_serving.model import opt_model_1d

from alpa.collective.worker_nccl_util_cupy import cupy_to_jax_tensor


def jax_to_cupy(jax_array):
    return cupy.from_dlpack(
        xc._xla.buffer_to_dlpack_managed_tensor(jax_array, take_ownership=False))


def init_1d_model(name, np_weights_folder, total_input_len, total_cache_len):
    # Init 1D model
    config = opt_model.get_opt_config(name, dtype=jnp.float32)
    model_1d, params_1d = opt_model_1d.init_model_aval_v2(config, total_input_len, total_cache_len)
    params_1d = opt_model_1d.load_params_np(params_1d, np_weights_folder,
                                            config)
    params_1d = jax.tree_map(jnp.array, params_1d)

    # Initialize the KV caches
    kv_caches = opt_model_1d.init_cache_np_v2(config, total_cache_len)
    kv_caches = [(jnp.asarray(k), jnp.asarray(v)) for k, v in kv_caches]
    kv_caches_cupy = [(jax_to_cupy(k), jax_to_cupy(v)) for k, v in kv_caches]

    @jax.jit
    def inference_step_1d(params, input_ids, position_ids, kv_caches):
        output = model_1d.apply(params,
                                input_ids,
                                position_ids,
                                attention_cache=kv_caches)
        return output.logits, output.attention_cache

    def runner(
        input_tokens: List[List[int]],
        input_sentence_ids: List[int],
        max_cache_len_per_seq: int,
        kv_cache_ids: np.array,
        num_prev_tokens: Dict[int, int],
    ) -> List[List[int]]:
        # input_tokens should not include paddings.
        for input_sentence in input_tokens:
            assert config.pad not in input_sentence

        # TODO: Reorder the input sentences.

        # Pre-update num_prev_tokens.
        for i, sentence_id in enumerate(input_sentence_ids):
            if sentence_id in num_prev_tokens:
                assert len(input_tokens[i]) == 1
                assert num_prev_tokens[sentence_id] > 0
            else:
                num_prev_tokens[sentence_id] = 0

        # Generate inputs.
        input_1d = sum(input_tokens, [])
        input_1d = input_1d + [config.pad] * (total_input_len - len(input_1d))
        input_1d = jnp.asarray(input_1d, dtype=jnp.int32)

        # Generate sentence ids.
        assert len(input_tokens) == len(input_sentence_ids)
        input_index_1d = []
        for i, sentence_id in enumerate(input_sentence_ids):
            input_index_1d.extend([sentence_id] * len(input_tokens[i]))
        input_index_1d = input_index_1d + [0] * (total_input_len - len(input_index_1d))
        input_index_1d = np.array(input_index_1d, dtype=np.int32)

        # Generate position ids.
        position_id_1d = []
        for i, sentence_id in enumerate(input_sentence_ids):
            start_idx = 1 + config.pad + num_prev_tokens[sentence_id]
            position_ids = list(range(start_idx, start_idx + len(input_tokens[i])))
            position_id_1d.extend(position_ids)
        position_id_1d = position_id_1d + [config.pad] * (total_input_len - len(position_id_1d))
        position_id_1d = jnp.asarray(position_id_1d, dtype=jnp.int32)

        assert max_cache_len_per_seq >= max(num_prev_tokens.values())
        assert max_cache_len_per_seq * len(num_prev_tokens) <= kv_cache_ids.shape[0]
        os.environ['FT_INPUT_INDEX_ADDR'] = str(input_index_1d.ctypes.data)
        os.environ['FT_CACHE_INDEX_ADDR'] = str(kv_cache_ids.ctypes.data)
        os.environ['FT_MAX_CACHE_LEN_PER_SEQ'] = str(max_cache_len_per_seq)

        # kv_caches = [(cupy_to_jax_tensor(k), cupy_to_jax_tensor(v)) for k, v in kv_caches_cupy]

        logits, kv = inference_step_1d(
            params_1d, input_1d, position_id_1d, kv_caches)

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
            key_1d = jax_to_cupy(key_1d)
            value_1d = jax_to_cupy(value_1d)
            idx = 0
            for i, sentence_id in enumerate(input_sentence_ids):
                # FIXME
                cache_idx = (sentence_id - 1) * max_cache_len_per_seq + num_prev_tokens[sentence_id]
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

        return outputs

    return runner

input_id_list = [
    # First batch
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

def test_opt_125M():
    name = "125M"
    np_weights_folder = f"/home/ubuntu/opt_weights/{name}_np"

    # Initialize the model.
    N = 64
    M = 512
    runner = init_1d_model(name, np_weights_folder, N, M)

    # Initialize the cache metadata.
    # These metadata are updated by the runner.
    kv_cache_ids = np.zeros((M,), dtype=np.int32)
    num_prev_tokens: Dict[int, int] = {} # sentence id -> number of previous tokens

    BATCH = 5
    output1 = runner(input_id_list[:BATCH], [1, 2, 3, 4, 5], 32, kv_cache_ids, num_prev_tokens)
    print(output1)
    inputs = input_id_list[BATCH:] + output1
    output2 = runner(inputs, [6, 7, 8, 9, 10] + [1, 2, 3, 4, 5], 32, kv_cache_ids, num_prev_tokens)
    print(output1, output2)

if __name__ == "__main__":
    test_opt_125M()

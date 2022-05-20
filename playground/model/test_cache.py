from functools import partial
import os

import jax
import jax.numpy as jnp
import numpy as np
from alpa.testing import assert_allclose

from opt_model import (OPTConfig, OPTForLMModule, init_model_aval, inference_step_no_cache,
                       build_init_cache, build_position_ids)

def load_params(params, path, num_layers):
    def load_array(key):
        return np.load(os.path.join(path, key))

    def load_param(param_key, loaded_array):
        param_dict = params
        param_keys = param_key.split('.')
        for i, key in enumerate(param_keys):
            if i == len(param_keys) - 1:
                assert param_dict[key].shape == loaded_array.shape
                assert param_dict[key].dtype == loaded_array.dtype
                param_dict[key] = loaded_array
            else:
                param_dict = param_dict[key]
    load_param("params.transformers.embeddings.word_embeddings.embedding",
               load_array("decoder.embed_tokens.weight"))
    load_param("params.transformers.embeddings.position_embeddings.embedding",
               load_array("decoder.embed_positions.weight"))
    for i in range(num_layers):
        param_prefix = f"params.transformers.encoder.{i}."
        load_prefix = f"decoder.layers.{i}."
        # Attention weights
        wq = load_array(load_prefix + "self_attn.q_proj.weight")
        wk = load_array(load_prefix + "self_attn.k_proj.weight")
        wv = load_array(load_prefix + "self_attn.v_proj.weight")
        w_qvk = np.transpose(np.concatenate([wq, wv, wk], axis=0))
        load_param(param_prefix + "attention.self.qvk_combined.kernel", w_qvk)
        bq = load_array(load_prefix + "self_attn.q_proj.bias")
        bk = load_array(load_prefix + "self_attn.k_proj.bias")
        bv = load_array(load_prefix + "self_attn.v_proj.bias")
        b_qvk = np.concatenate([bq, bv, bk], axis=0)
        load_param(param_prefix + "attention.self.qvk_combined.bias", b_qvk)
        load_param(param_prefix + "attention.dense.kernel",
                   np.transpose(load_array(load_prefix + "self_attn.out_proj.weight")))
        load_param(param_prefix + "attention.dense.bias",
                   load_array(load_prefix + "self_attn.out_proj.bias"))
        load_param(param_prefix + "attention.layer_norm.scale",
                   load_array(load_prefix + "self_attn_layer_norm.weight"))
        load_param(param_prefix + "attention.layer_norm.bias",
                   load_array(load_prefix + "self_attn_layer_norm.bias"))
        # FFN weights
        load_param(param_prefix + "ffn.fc1.bias",
                   load_array(load_prefix + "fc1.bias"))
        load_param(param_prefix + "ffn.fc1.kernel",
                   np.transpose(load_array(load_prefix + "fc1.weight")))
        load_param(param_prefix + "ffn.fc2.bias",
                   load_array(load_prefix + "fc2.bias"))
        load_param(param_prefix + "ffn.fc2.kernel",
                   np.transpose(load_array(load_prefix + "fc2.weight")))
        load_param(param_prefix + "ffn.layer_norm.scale",
                   load_array(load_prefix + "final_layer_norm.weight"))
        load_param(param_prefix + "ffn.layer_norm.bias",
                   load_array(load_prefix + "final_layer_norm.bias"))
    return params


def print_params(params, prefix=""):
    for key, value in params.items():
        if isinstance(value, dict):
            print_params(value, prefix=prefix + key + ".")
        else:
            print(prefix + key, value.shape)


def test_opt_125M():
    #TODO: align dtype
    config = OPTConfig()
    numpy_weights_folder = "./numpy_weights"

    # Init model
    input_ids = jnp.array([[5625,   16,   10, 2721,  183,    8,   38,  236,    7]], dtype=jnp.int32)
    position_ids = build_position_ids(input_ids, config.pad)
    print("input_ids", input_ids)

    model, params = init_model_aval(config)
    params = load_params(params.unfreeze(), numpy_weights_folder, num_layers=config.decoder_layers)

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

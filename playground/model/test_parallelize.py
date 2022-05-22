import dataclasses
from functools import partial
import os

import jax
import jax.numpy as jnp
import numpy as np
from alpa.testing import assert_allclose
import alpa

from opt_model import (OPTConfig, OPTForLMModule, init_model_aval, inference_step_no_cache,
                       build_init_cache, build_position_ids, load_np_params)

def test_opt_125M_shard_parallel():
    #TODO: align dtype
    config = OPTConfig()
    numpy_weights_folder = "./numpy_weights"

    # Init model and optimizer
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

    # Parallelize
    method = alpa.ShardParallel(
        devices=jax.local_devices()[:4],
        auto_sharding_option=alpa.AutoShardingOption())

    @alpa.parallelize(static_argnums=(2,), batch_argnums=(), method=method)
    def inference_step_with_cache(params, batch, apply_func):
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

    # Dump IR
    executable = inference_step_with_cache.last_executable
    with open("infer.hlo", "w") as fout:
        fout.write(executable.get_hlo_text())

    assert executable.get_hlo_text().count("all-reduce(") == 1 + 2 * config.decoder_layers


def test_opt_125M_pipeshard_parallel():
    #TODO: align dtype
    config = OPTConfig()
    config = dataclasses.replace(config, num_pp_stages=2)
    numpy_weights_folder = "./numpy_weights"
    
    alpa.init()

    # Init model and optimizer
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

    # Parallelize
    method = alpa.PipeshardParallel(num_micro_batches=1,
                                    pipeline_schedule="inference")

    @alpa.parallelize(static_argnums=(3,), batch_argnums=(1,), method=method)
    def inference_step_with_cache(params, batch, cache, apply_func):
        @alpa.manual_layer_construction
        def forward(params, cache):
            alpa.mark_pipeline(name="0", mark_type="start")
            output = apply_func(params,
                                batch["input_ids"],
                                batch["position_ids"],
                                attention_cache=cache)
            alpa.mark_pipeline(name=f"{config.num_pp_stages - 1}", mark_type="end")
            return output

        output = forward(params, cache)
        return output.logits, output.attention_cache

    cache = build_init_cache(config)

    for i in range(input_ids.shape[1]):
        input_ids_step = input_ids[:, i:i+1]
        position_ids_step = jnp.full_like(input_ids_step, i + config.pad + 1)
        logits_step, cache = inference_step_with_cache(params, {
            "input_ids": input_ids_step,
            "position_ids": position_ids_step,
        }, cache, model.apply)
        assert_allclose(logits_step, logits_no_cache[:, i:i+1])

    # Dump IR
    executable = inference_step_with_cache.last_executable
    os.system("mkdir -p tmp")
    stage_hlo_texts = executable.get_hlo_text()
    for i in range(len(stage_hlo_texts)):
        with open(f"tmp/stage_{i}.hlo", "w") as fout:
            fout.write(stage_hlo_texts[i])
    with open(f"tmp/resharding_tasks.txt", "w") as fout:
        fout.write(executable.print_resharding_tasks())


if __name__ == "__main__":
    #test_opt_125M_shard_parallel()
    test_opt_125M_pipeshard_parallel()

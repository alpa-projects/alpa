import dataclasses
from functools import partial
import os

import jax
import jax.numpy as jnp
import numpy as np
from alpa.testing import assert_allclose
import alpa

from examples.opt_serving.model.opt_model import (get_config, init_model_aval,
                                                  inference_step_no_cache,
                                                  init_cache_np,
                                                  build_position_ids,
                                                  load_params_np)


def test_opt_125M_shard_parallel():
    name = "125M"
    config = get_config(name)
    np_weights_folder = f"/home/ubuntu/opt_weights/{name}_np"

    # Init model
    input_ids = jnp.array([[5625, 16, 10, 2721, 183, 8, 38, 236, 7]],
                          dtype=jnp.int32)
    position_ids = build_position_ids(input_ids, config.pad)
    print("input_ids", input_ids)

    model, params = init_model_aval(config, jnp.float16)
    params = load_params_np(params, np_weights_folder, config)

    # Get expected results
    logits_no_cache = inference_step_no_cache(params, {
        "input_ids": input_ids,
        "position_ids": position_ids,
    }, model.apply)
    print("logits_no_cache", logits_no_cache)

    # Parallelize
    method = alpa.ShardParallel(devices=jax.local_devices()[:4],
                                auto_sharding_option=alpa.AutoShardingOption())

    @alpa.parallelize(method=method)
    def inference_step_with_cache(params, batch):
        output = model.apply(params,
                             batch["input_ids"],
                             batch["position_ids"],
                             attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    cache = build_init_cache(config)

    for i in range(input_ids.shape[1]):
        input_ids_step = input_ids[:, i:i + 1]
        position_ids_step = jnp.full_like(input_ids_step, i + config.pad + 1)
        logits_step, cache = inference_step_with_cache(
            params, {
                "input_ids": input_ids_step,
                "position_ids": position_ids_step,
                "cache": cache,
            })
        assert_allclose(logits_step, logits_no_cache[:, i:i + 1])

    # Dump IR
    executable = inference_step_with_cache.last_executable
    with open("infer.hlo", "w") as fout:
        fout.write(executable.get_hlo_text())

    assert executable.get_hlo_text().count(
        "all-reduce(") == 1 + 2 * config.decoder_layers


def test_opt_125M_pipeshard_parallel():
    name = "125M"
    config = get_config(name, num_pp_stages=2)
    np_weights_folder = f"/home/ubuntu/opt_weights/{name}_np"
    batch_size = 2

    alpa.init()

    # Init model and optimizer
    input_ids = jnp.array([[5625, 16, 10, 2721, 183, 8, 38, 236, 7]],
                          dtype=jnp.int32)
    input_ids = np.tile(input_ids, [batch_size, 1])
    position_ids = build_position_ids(input_ids, config.pad)
    print("input_ids", input_ids)

    model, params = init_model_aval(config)
    params = load_params_np(params, np_weights_folder, config)

    # Get expected results
    logits_no_cache = inference_step_no_cache(params, {
        "input_ids": input_ids,
        "position_ids": position_ids,
    }, model.apply)
    print("logits_no_cache", logits_no_cache)

    # Parallelize
    method = alpa.PipeshardParallel(num_micro_batches=1,
                                    pipeline_schedule="inference")

    @alpa.parallelize(method=method)
    def inference_step_with_cache(params, batch):

        @alpa.manual_layer_construction
        def forward(params, cache):
            alpa.mark_pipeline(name="0", mark_type="start")
            output = model.apply(params,
                                 batch["input_ids"],
                                 batch["position_ids"],
                                 attention_cache=batch["cache"])
            alpa.mark_pipeline(name=f"{config.num_pp_stages - 1}",
                               mark_type="end")
            return output

        output = forward(params, cache)
        return output.logits, output.attention_cache

    cache = init_cache_np(config, batch_size)

    for i in range(input_ids.shape[1]):
        input_ids_step = input_ids[:, i:i + 1]
        position_ids_step = jnp.full_like(input_ids_step, i + config.pad + 1)
        logits_step, cache = inference_step_with_cache(
            params, {
                "input_ids": input_ids_step,
                "position_ids": position_ids_step,
                "cache": cache,
            })
        assert_allclose(logits_step, logits_no_cache[:, i:i + 1])

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

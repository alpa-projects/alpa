"""Benchmark one case of inter-op + intra-op parallelism."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

import alpa
from alpa import (parallelize, global_config, get_global_cluster,
                  set_global_virtual_physical_mesh, AutoShardingOption,
                  PipeshardParallel, ManualStageOption, AutoStageOption,
                  AutoLayerOption)
from alpa.model.model_util import optax_adafactor
from alpa.model.moe import FlaxMoEForLMModule, MoEConfig, TrainState
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.timer import timers
from alpa.util import print_used_time, to_str_round, GB

from benchmark_3d_one_case_gpt_bert import (get_train_step,
                                            compile_and_benchmark_executable)
from benchmark.util import compute_moe_parameter_count, compute_moe_tflops
from parallel_option import get_parallel_method


def create_train_state(rngkey, model, dtype, batch):
    params = model.init_dummy(rngkey, batch["input_ids"],
                              batch["attention_mask"], batch["token_type_ids"],
                              batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax_adafactor(learning_rate=1e-2,
                         weight_decay_mask=weight_decay_mask)

    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              mixed_precision=(dtype == jnp.float16),
                              dynamic_scale=None)
    return state


def prepare_moe_input_and_model(benchmark_case,
                                add_manual_remat=None,
                                add_manual_layer_marker=None,
                                num_manual_pipeline_stages=None):
    print_used_time(None)
    (batch_size, model_config, num_micro_batches, parallel_mode,
     parallel_args) = benchmark_case
    (seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts,
     expert_group_size) = model_config
    dtype = jnp.float16
    tie_word_embeddings = False

    rang_factor = 1
    expected_expert_group_size = min(
        expert_group_size,
        batch_size * seq_len // num_micro_batches // 1 // rang_factor)
    if expected_expert_group_size != expert_group_size:
        print("- Expected expert group size should be {}, "
              "but got {}. Will reset it".format(expected_expert_group_size,
                                                 expert_group_size))
        expert_group_size = expected_expert_group_size

    # Prepare input batch
    batch = {
        "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "token_type_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "position_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
    }
    print_used_time("Prepare input")

    # Init train state
    model = FlaxMoEForLMModule(
        MoEConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 8,  # this is specific to gspmd.
            num_attention_heads=num_heads,
            max_position_embeddings=seq_len,
            vocab_size=vocab_size,
            expert_group_size=expert_group_size,
            expert_number=num_experts,
            tie_word_embeddings=tie_word_embeddings,
            gradient_checkpointing=add_manual_remat,
            add_manual_pipeline_markers=add_manual_layer_marker,
            pipeline_mp_size=num_manual_pipeline_stages,
        ),
        dtype=dtype)

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, dtype, batch)
    print_used_time("Create train state")
    return state, batch, rngkey


def compute_moe_statistics(benchmark_case, latencies, num_devices):
    (batch_size, model_config, num_micro_batches, parallel_mode,
     parallel_args) = benchmark_case
    (seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts,
     expert_group_size) = model_config
    tflops = compute_moe_tflops(batch_size, seq_len, num_layers, hidden_size,
                                expert_group_size, vocab_size, num_experts,
                                num_devices, np.mean(latencies))
    tflops_ckpt = compute_moe_tflops(batch_size,
                                     seq_len,
                                     num_layers,
                                     hidden_size,
                                     expert_group_size,
                                     vocab_size,
                                     num_experts,
                                     num_devices,
                                     np.mean(latencies),
                                     checkpoint_activations=True)
    parameter_count = compute_moe_parameter_count(num_layers,
                                                  hidden_size,
                                                  vocab_size,
                                                  num_experts,
                                                  mlp_factor=8)
    return tflops, tflops_ckpt, parameter_count


def benchmark_moe_internal(benchmark_case, niter, num_hosts,
                           num_devices_per_host):
    # Connect to the cluster
    virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
        host_ids=list(range(num_hosts)),
        num_devices_per_host=num_devices_per_host)
    set_global_virtual_physical_mesh(virtual_mesh)

    # Parallel configs
    if benchmark_case.parallel_mode == "load_solution":
        use_fine_grained_remat = benchmark_case.parallel_args.use_remat
        fine_grained_remat_num_layers = benchmark_case.model_config.num_layers
    else:
        use_fine_grained_remat = None
        fine_grained_remat_num_layers = None
    (method, add_manual_remat, add_manual_layer_marker,
     num_manual_pipeline_stages) = get_parallel_method(
         benchmark_case,
         virtual_mesh.num_devices_per_host,
         allow_mixed_mesh_shape=True,
         use_fine_grained_remat=use_fine_grained_remat)

    state, batch, rngkey = prepare_moe_input_and_model(
        benchmark_case,
        add_manual_remat=add_manual_remat,
        add_manual_layer_marker=add_manual_layer_marker,
        num_manual_pipeline_stages=num_manual_pipeline_stages)

    train_step = get_train_step(method, use_fine_grained_remat,
                                fine_grained_remat_num_layers)

    (latencies, max_mem_allocated, compilation_times,
     executable) = compile_and_benchmark_executable(
         benchmark_case.parallel_mode, niter, train_step, state,
         (batch, rngkey))

    tflops, tflops_ckpt, parameter_count = compute_moe_statistics(
        benchmark_case, latencies, virtual_mesh.num_devices)

    return (parameter_count, max_mem_allocated, latencies, tflops, tflops_ckpt,
            compilation_times) + get_last_dp_result()

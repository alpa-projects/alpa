"""Benchmark one case of inter-op + intra-op parallelism."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

import alpa
from alpa import (parallelize, global_config, get_global_cluster,
                  set_global_virtual_physical_mesh, AutoShardingOption,
                  PipeshardParallel, ManualPipeshardParallel)
from alpa.model.model_util import optax_adafactor
from alpa.model.moe import FlaxMoEForLMModule, MoEConfig, TrainState
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.timer import timers
from alpa.util import print_used_time, to_str_round, GB

from benchmark_3d_one_case_gpt_bert import get_train_step
from benchmark.util import compute_moe_parameter_count, compute_moe_tflops


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


def benchmark_moe_internal(benchmark_case, niter, num_hosts,
                           num_devices_per_host):
    print_used_time(None)

    # Model configs
    (batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,
     num_experts, expert_group_size, num_micro_batches, parallel_mode,
     parallel_args) = benchmark_case
    dtype = jnp.float16
    tie_word_embeddings = False

    rang_factor = 1
    expected_expert_group_size = min(
        expert_group_size,
        batch_size * seq_len // num_micro_batches // 1 // rang_factor)
    if expected_expert_group_size != expert_group_size:
        print(
            "- Expected expert group size should be {}, but got {}. Will reset it"
            .format(expected_expert_group_size, expert_group_size))
        expert_group_size = expected_expert_group_size

    # Connect to the cluster
    virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
        host_ids=list(range(num_hosts)),
        num_devices_per_host=num_devices_per_host)
    set_global_virtual_physical_mesh(virtual_mesh)

    # Parallel configs
    if parallel_mode == "search":
        prefer_reduce_scatter, use_remat, num_auto_layers, auto_stage_option = parallel_args
        auto_layer = True
        auto_remat_mode = "fine_grained" if use_remat else None
        num_auto_remat_layers = num_layers
        add_manual_layer_marker = add_manual_remat = num_manual_pipeline_stages = False
        method = PipeshardParallel(
            stage_mode="auto",
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter,
                allow_mixed_mesh_shape=True,
            ),
            **auto_stage_option)
    elif parallel_mode == "load_solution":
        prefer_reduce_scatter, use_remat, num_auto_layers, manual_stage_option = parallel_args
        auto_layer = True
        auto_remat_mode = "fine_grained" if use_remat else None
        num_auto_remat_layers = num_layers
        add_manual_layer_marker = add_manual_remat = num_manual_pipeline_stages = False
        method = ManualPipeshardParallel(
            *manual_stage_option,
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter,
                allow_mixed_mesh_shape=True,
            ))
    elif parallel_mode == "manual":
        (prefer_reduce_scatter, use_remat, (dp, op, pp),
         force_batch_dim_mapping) = parallel_args
        as_option = AutoShardingOption(
            prefer_reduce_scatter=prefer_reduce_scatter)
        if force_batch_dim_mapping:
            as_option.force_batch_dim_to_mesh_dim = 0
        auto_layer = False
        num_auto_layers = auto_remat_mode = num_auto_remat_layers = None
        add_manual_layer_marker = True
        add_manual_remat = use_remat

        logical_mesh_shape = (dp, op)
        num_manual_pipeline_stages = pp
        num_mesh_devices = np.prod(logical_mesh_shape)
        num_devices_per_host = virtual_mesh.num_devices_per_host
        if num_mesh_devices <= num_devices_per_host:
            physical_mesh_shape = (1, num_mesh_devices)
        else:
            assert num_mesh_devices % num_devices_per_host == 0
            physical_mesh_shape = (num_mesh_devices // num_devices_per_host,
                                   num_devices_per_host)

        method = ManualPipeshardParallel(
            num_micro_batches=num_micro_batches,
            forward_stage_layer_ids=[[i] for i in range(pp)],
            submesh_physical_shapes=[physical_mesh_shape] * pp,
            submesh_logical_shapes=[logical_mesh_shape] * pp,
            submesh_autosharding_option_dicts=[{}] * pp,
            default_auto_sharding_option=as_option)
    else:
        raise ValueError(f"Invalid model: {parallel_mode}")

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

    # Compile executable
    train_step = get_train_step(method, auto_layer, num_manual_pipeline_stages,
                                num_auto_layers, auto_remat_mode,
                                num_auto_remat_layers)
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    if parallel_mode == "search":
        compilation_times = {
            k: timers(k).elapsed() for k in [
                "stage-construction", "stage-construction-dp",
                "stage-construction-compilation", "stage-construction-profiling"
            ]
        }
        print(
            f"compilation time breakdown: {to_str_round(compilation_times, 2)}")
    else:
        compilation_times = None

    executable.dump_debug_info("tmp")
    executable.sync()
    print_used_time("Compile (worker)")

    # Benchmark step time
    for i in range(niter):
        print(f"Iteration: {i} ...")
        state = train_step(state, batch, rngkey)
        executable.sync()

    latencies = executable.get_execution_time_costs()[1:]
    max_mem_allocated = executable.mesh_group.get_max_memory_allocated()
    print_used_time("Benchmark")

    # Compute statistics
    tflops = compute_moe_tflops(batch_size, seq_len, num_layers, hidden_size,
                                expert_group_size, vocab_size, num_experts,
                                virtual_mesh.num_devices, np.mean(latencies))
    tflops_ckpt = compute_moe_tflops(batch_size,
                                     seq_len,
                                     num_layers,
                                     hidden_size,
                                     expert_group_size,
                                     vocab_size,
                                     num_experts,
                                     virtual_mesh.num_devices,
                                     np.mean(latencies),
                                     checkpoint_activations=True)
    parameter_count = compute_moe_parameter_count(num_layers,
                                                  hidden_size,
                                                  vocab_size,
                                                  num_experts,
                                                  mlp_factor=8)

    return (parameter_count, max_mem_allocated, latencies, tflops, tflops_ckpt,
            compilation_times) + get_last_dp_result()

"""Benchmark one case of intra-op only parallelism."""
import jax
import jax.numpy as jnp
import numpy as np
import ray

import alpa
from alpa import global_config, set_parallelize_options
from alpa.model.moe import FlaxMoEForLMModule, MoEConfig, TrainState
from alpa.model.model_util import optax_adafactor
from alpa.util import count_communication_primitives, print_used_time, compute_param_number, GB

from benchmark.util import compute_moe_parameter_count
from benchmark_2d_one_case_gpt_bert import get_train_step

as_option = global_config.default_autosharding_option


def create_train_state(rngkey, model, dtype, batch):
    params = model.init_dummy(rngkey, batch["input_ids"], batch["attention_mask"],
                              batch["token_type_ids"], batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax_adafactor(
        learning_rate=1e-2, weight_decay_mask=weight_decay_mask
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        mixed_precision = (dtype == jnp.float16),
        dynamic_scale=None)
    return state


def benchmark_moe_internal(physical_mesh, benchmark_case, niter):
    print_used_time(None)

    # Model configs
    (batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts,
     expert_group_size, num_micro_batches, parallel_mode, parallel_args) = benchmark_case
    (prefer_reduce_scatter, use_remat, (dp, op, pp),
     force_batch_dim_mapping) = parallel_args
    dtype = jnp.float16

    #rang_factor = 1
    #expected_expert_group_size = min(expert_group_size, batch_size * seq_len // num_micro_batches // mesh_dim0 // rang_factor)
    #if expected_expert_group_size != expert_group_size:
    #    print("- Expected expert group size should be {}, but got {}. Will reset it".
    #          format(expected_expert_group_size, expert_group_size))
    #    expert_group_size = expected_expert_group_size

    # Parallel configs
    if num_micro_batches > 1:
        grad_func = alpa.grad
    else:
        num_micro_batches = None
        grad_func = jax.grad

    if force_batch_dim_mapping:
        # Always map batch dim to mesh dim 0
        as_option.force_batch_dim_to_mesh_dim = 0
    as_option.prefer_reduce_scatter = prefer_reduce_scatter
    as_option.allow_mixed_mesh_shape = True

    if parallel_mode == "zero-3":
        as_option.force_zero_stage_3 = True
    elif parallel_mode in ["shard-largest"]:
        as_option.force_simple_heuristic = other
        global_config.remat_using_while = True


    logical_mesh = physical_mesh.get_logical_mesh([dp, op])
    set_parallelize_options(devices=logical_mesh, num_micro_batches=num_micro_batches)

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
    model = FlaxMoEForLMModule(MoEConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 8,
        num_attention_heads=num_heads,
        max_position_embeddings=seq_len,
        vocab_size=vocab_size,
        expert_group_size=expert_group_size,
        expert_number=num_experts,
        gradient_checkpointing=use_remat
    ), dtype=dtype)

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, dtype, batch)
    param_count = compute_param_number(state.params)
    print_used_time("Create train state")

    # Compile executable
    train_step = get_train_step(grad_func)
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    physical_mesh.sync_workers()
    print_used_time("Compile (workers)")

    # Check sharding strategy
    alloc_mem = executable.get_total_allocation_size()
    ilp_objective = executable.auto_sharding_objective or 0.0
    hlo_text = executable.get_hlo_text()
    with open("tmp/last_2d_moe.hlo", "w") as fout:
        fout.write(hlo_text)
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all =\
        count_communication_primitives(hlo_text)

    print(f"#total: {n_total}, #all-reduce: {n_all_reduce}, "
          f"#all-gather: {n_all_gather}, #reduce-scatter: {n_reduce_scatter}, "
          f"#all-to-all: {n_all_to_all}")
    print(f"alloc_mem: {alloc_mem / GB:.2f} GB")

    # Benchmark step time
    warmup = 2 if niter >= 5 else 1

    if alloc_mem > physical_mesh.get_available_memory():
        latencies = [-1]
    else:
        for i in range(niter):
            print(f"Iteration {i} ...")
            state = train_step(state, batch, rngkey)

        latencies = executable.get_execution_time_costs(warmup=warmup)
    print_used_time("Benchmark")

    # Compute statistics
    num_gpus = physical_mesh.num_devices
    tflops = executable.flop_count / num_gpus / np.mean(latencies) / 1e12
    parameter_count = compute_moe_parameter_count(num_layers, hidden_size, vocab_size, num_experts,
                                                  mlp_factor=8)
    peak_mem = max(physical_mesh.get_max_memory_allocated(), alloc_mem)

    return parameter_count, ilp_objective, peak_mem, latencies, tflops

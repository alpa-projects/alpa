import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

import parax
from benchmark_gpt_bert_3d import get_train_step
from benchmark.util import compute_moe_parameter_count, compute_moe_tflops
from parax import (global_config, set_parallelize_options, DeviceCluster)
from parax.model.model_util import optax_adafactor
from parax.model.moe import FlaxMoEForLMModule, MoEConfig, TrainState
from parax.util import (write_tsv, print_used_time, compute_param_number)

GB = 1024 ** 3


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


def benchmark_one_case(benchmark_case):
    # Backup global config
    backup = global_config.backup()
    print_used_time(None)

    model_type = "MoE"
    # Model configs
    batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,\
        expert_group_size, expert_number,\
        l_dim0, l_dim1, p_dim0, p_dim1, pipeline_mp_size, num_micro_batches, \
        force_data_parallel, use_remat = benchmark_case
    dtype = jnp.float16
    tie_word_embeddings = False
    prefer_reduce_scatter = False

    # Parallel configs
    grad_func = parax.grad

    global_config.force_data_parallel = force_data_parallel
    global_config.prefer_reduce_scatter = prefer_reduce_scatter

    device_cluster = DeviceCluster()
    virtual_mesh = device_cluster.get_virtual_mesh()
    set_parallelize_options(devices=virtual_mesh,
                            strategy="3d_parallel",
                            num_micro_batches=num_micro_batches,
                            sub_physical_mesh_shapes=[(p_dim0, p_dim1)] * pipeline_mp_size,
                            sub_logical_mesh_shapes=[(l_dim0, l_dim1)] * pipeline_mp_size)

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
        intermediate_size=hidden_size * 8, # this is specific to gspmd.
        num_attention_heads=num_heads,
        max_position_embeddings=seq_len,
        vocab_size=vocab_size,
        expert_group_size=expert_group_size,
        expert_number=expert_number,
        pipeline_mp_size=pipeline_mp_size,
        tie_word_embeddings=tie_word_embeddings
    ), dtype=dtype)

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, dtype, batch)
    print_used_time("Create train state")

    # Compile executable
    train_step = get_train_step(grad_func, num_layers, use_remat, pipeline_mp_size, dtype)
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")


    for i in range(args.niter):
        state = train_step(state, batch, rngkey)

    param_count = compute_moe_parameter_count(num_layers, hidden_size, vocab_size, expert_number,
                                              tie_embedding=tie_word_embeddings)

    timer_name = "overall"
    overall_costs = executable.get_execution_time_costs(warmup=0, timer_name=timer_name)
    print_used_time("Benchmark")

    # Compute statistics
    tflops = compute_moe_tflops(batch_size, seq_len, num_layers,
                                hidden_size, vocab_size, expert_number,
                                virtual_mesh.total_devices, np.mean(overall_costs[2:]))

    # Restore global config
    global_config.restore(backup)


    # Log results
    heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape", "#Microbatch",
             "Force DP", "Remat", "Mean Time", "Std Time", "#Params", "TFLOPs"]
    paralell_config = (benchmark_case[8], benchmark_case[9], benchmark_case[12])
    values = [model_type, str(benchmark_case[:8]), str(paralell_config), str(benchmark_case[10:12]),
              str(benchmark_case[13]), str(benchmark_case[14]), str(benchmark_case[15]),
              f"{np.mean(overall_costs[2:]):.3f}", f"{np.std(overall_costs[2:]):.3f}",
              str(param_count), str(tflops)]
    write_tsv(heads, values, f"result_{model_type}.tsv")

    executable.shutdown()


# B = batch_size, S = seq_len, H = hidden_size, L = num_layers,
# #head = num_heads, V= vocab_size, S_ = expert capacity, E = expert_number,
# LD0 = logical mesh dim 0, LD1 = logical mesh_dimension_1
# PD0 = physical mesh dim 0, PD1 = physical mesh dim 1
# FD = Force DP, NB = number of microbatches, Remat: rematerialization


sanity_check_suite = {

4: [
    # B,  S,     H,    L,  #head,    V,     S_,   E,  LD0, LD1, PD0, PD1, PP, NB,   FD,  Remat, (Tie)
    (8, 1024,    512,  2,  128,      1000,  1024, 4,  2,   1,   1,   2,   2,  2,   True, True),
],

8: [
    # B,  S,     H,    L,  #head,    V,     S_,   E,  LD0, LD1, PD0, PD1, PP, NB,   FD,  Remat, (Tie)
    (8,  1024,  512,   2, 128,      1000, 1024,  4,  4,   1,   1,   4,   2,  2,    True, True),
]

}

default_benchmark_suite = {  # key = number of gpus, value = a list of cases
1: [
],

2: [
],

8: [
    # B,  S,     H,    L,  #head,    V,     S_,   E,  LD0, LD1, PD0, PD1, PP, NB,   FD,  Remat, (Tie)
    (32,  1024,  8192, 10, 128,      32000, 1024, 4,  4,   1,   1,   4,   2,  1,    True, True),
    (32,  1024,  8192, 10, 128,      32000, 1024, 4,  4,   1,   1,   4,   2,  1,    True, True),
    (32,  1024,  8192, 10, 128,      32000, 1024, 4,  4,   1,   1,   4,   2,  1,    True, True),
    (32,  1024,  8192, 10, 128,      32000, 1024, 4,  4,   1,   1,   4,   2,  1,    True, True),

    # (16,  1024, 2560, 12, 2560//128, 25600, 1024, 16, 1,  8,  1,  False, False, False),
    # (16,  1024, 2560, 12, 2560//128, 25600, 1024, 16, 1,  8,  1,  True,  True,  False),
],

16: [
]
}


gspmd_suite = {

64: [
    # B,  S,     H,    L,  #head,    V      LD0, LD1, PD0, PD1, PP, NB,   FD,  Remat, Tie
    (512,   1024,  8192, 24, 128,      32000, 8,   1,   8,   1,   4,  64, True, True, True)
]


}

benchmark_suites = {
    "default": default_benchmark_suite,
    "sanity_check": sanity_check_suite
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--niter", type=int, default=10,
        help="Number of benchmark iteration")
    parser.add_argument("--suite", choices=["default", "sanity_check"], default="sanity_check")
    args = parser.parse_args()

    # Set global environments

    ray.init(address="auto")
    jax.config.update('jax_platform_name', 'cpu')
    num_gpus = int(ray.cluster_resources()["GPU"])

    global_config.use_dummy_value_for_benchmarking = True

    # Get benchmark suite and run all cases
    try:
        suite = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        suite = None

    if not suite:
        print(f"No available benchmark suite for {args.suite} on {num_gpus} GPUs")
        exit()

    for case in suite:
        benchmark_one_case(case)

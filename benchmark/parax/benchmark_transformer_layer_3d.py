import argparse

import copy
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray
from functools import partial

import parax
from parax import (parallelize, global_config, set_parallelize_options, DeviceCluster,
                   mark_pipeline, manual_pipeline, forward)
from parax.model.bert_model import BertConfig, FlaxBertLayerCollection, TrainState
from parax.util import write_tsv, list_gpu_info, print_used_time

MB = 1024 ** 2
GB = 1024 ** 3


def create_train_state(rngkey, model, batch):
    params = model.init_dummy(rngkey, batch["hidden_states"], batch["attention_mask"])
    tx = optax.adam(learning_rate=1e-2)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dynamic_scale=None)
    return state


def get_train_step(grad_func, num_layers, use_remat, dtype, pipeline_mp_size):

    @parallelize
    def train_step(state, batch, rng_key):
        @partial(forward, layer_num=num_layers, use_remat=use_remat)
        def loss_func(params):
            rngs = {"dropout": rng_key}
            if pipeline_mp_size > 1:
                mark_pipeline(name="0", mark_type="start")
            out = state.apply_fn(params,
                                 batch["hidden_states"],
                                 batch["attention_mask"],
                                 deterministic=True,
                                 rngs=rngs)[0]
            loss = jnp.mean((out - batch["label"]) ** 2)
            if pipeline_mp_size > 1:
                mark_pipeline(name=str(pipeline_mp_size - 1), mark_type="end")
            return loss
        if pipeline_mp_size > 1:
            loss_func = manual_pipeline(loss_func)
        # grad, grad_x = jax.grad(loss_func, argnums=(0, 1))(optimizer.target, batch["hidden_states"])
        grad = grad_func(loss_func, argnums=(0))(state.params)
        # new_state = state.apply_gradients(grads=grads)
        return grad

    return train_step


def benchmark_transformer_one_case(benchmark_case):
    print_used_time(None)

    # Model configs
    batch_size, seq_len, hidden_size, num_layers, num_heads, \
    mesh_dim0, mesh_dim1, pipeline_mp_size, num_micro_batches, force_data_parallel, \
    use_remat = benchmark_case
    dtype = jnp.float16

    global_config.force_data_parallel = force_data_parallel
    global_config.prefer_reduce_scatter = False
    if num_micro_batches > 1:
        grad_func = parax.grad
    else:
        grad_func = jax.grad

    # Mesh configs
    # 3D parallel always run atop a Ray cluster.
    device_cluster = DeviceCluster()
    virtual_mesh = device_cluster.get_virtual_mesh()
    set_parallelize_options(devices=virtual_mesh,
                            strategy="3d_parallel",
                            num_micro_batches=num_micro_batches)

    # Prepare input batch
    batch = {
        "hidden_states": jnp.ones((batch_size, seq_len, hidden_size), dtype=np.float32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=np.int32),
        "label": jnp.ones((batch_size, seq_len, hidden_size), dtype=np.float32),
    }
    print_used_time("Prepare input")

    # Init model and optimizer
    model = FlaxBertLayerCollection(BertConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=num_heads,
        pipeline_mp_size=pipeline_mp_size))

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, batch)
    print_used_time("Create train state")

    # Compile executable
    train_step = get_train_step(grad_func, num_layers, use_remat, dtype, pipeline_mp_size)
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")


    for i in range(args.niter):
        train_step(state, batch, rngkey)

    costs = executable.get_execution_time_costs(warmup=2)
    print_used_time("Benchmark")

    print(costs)
    # Log benchmark results
    heads = ["Type", "Model Config", "Parallel Config", "# Microbatch", "Mean Time", "Std Time"]
    values = ["transformer-layer", str(benchmark_case[:5]), str(benchmark_case[5:]),
             f"{benchmark_case[8]:.3f}", f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}"]
    write_tsv(heads, values, "result_trans.tsv")

    executable.shutdown()

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers,
# #head = num_heads, D0 = mesh_dimension_0, D1 = mesh_dimension_1

benchmark_suite_4_gpu = [
    # # B,  S,    H,    L,  #head,     D0, D1, PP, NB, FD, CK
    (32,  1024, 1536, 2,  1536//96,  2,  1, 2, 1, False, False),
    (32,  1024, 1536, 2,  1536//96,  1,  2, 2, 2, False, False),
    (32,  128,  5120, 2,  5120//128, 1,  2, 2, 4, False, False),
    (32,  128,  5120, 2,  5120//128, 2,  1, 2, 8, False, False),
]

benchmark_suite_8_gpu = [
    # B,  S,    H,    L,  #head,     D0, D1, PP, NB, FD, CK
    # (32,  1024, 1536, 2,  1536//96,  4,  1, 2, 1, False),
    (16,  1024, 1536, 2,  1536//96,  4,  1, 2, 1, True, False),
    #
    # (32,  128,  5120, 2,  5120//128, 1,  4, 2, 1, False),
    # (32,  128,  5120, 2,  5120//128, 4,  1, 2, 1, False),
]


def benchmark_all(use_profiling):
    if args.local:
        num_gpus = list_gpu_info().count("UUID")
    else:
        num_gpus = int(ray.cluster_resources()["GPU"])

    benchmark_suites = {
        4: benchmark_suite_4_gpu,
        8: benchmark_suite_8_gpu,
    }

    for case in benchmark_suites[num_gpus]:
        # Backup global config
        backup = global_config.backup()
        benchmark_transformer_one_case(case)
        # Restore global config
        global_config.restore(backup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--number", type=int, default=5)
    parser.add_argument("--local", action="store_true",
                        help="Run on local GPUs. Do not use ray actors.")
    parser.add_argument("--niter", type=int, default=10,
                        help="Number of benchmark iteration")
    args = parser.parse_args()

    if not args.local:
        ray.init(address="auto")
        jax.config.update('jax_platform_name', 'cpu')

    global_config.use_dummy_value_for_benchmarking = True

    benchmark_all(args.use_profiling)

import argparse
from functools import partial
import time

from flax import linen as nn, optim
from flax.core.frozen_dict import FrozenDict as FrozenDictFlax
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import ray
import optax

import alpa
from alpa import (parallelize, global_config, set_parallelize_options, testing,
                   DeviceCluster, LocalPhysicalDeviceMesh, automatic_layer_construction)
from alpa.model.conformer import ConformerForASRModule, ConformerConfig, TrainState
from alpa.util import (run_cmd, write_tsv, map_to_shape, list_gpu_info,
                        count_communication_primitives, print_used_time,
                        compute_param_number)


GB = 1024 ** 3


def create_train_state(rngkey, model, batch):
    params = model.init_dummy(rngkey, batch["input_frames"], batch["attention_mask"])
    params, batch_stats = params["params"], params.get("batch_stats", FrozenDictFlax())

    tx = optax.chain(
        optax.adam(learning_rate=1e-3)
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        dynamic_scale=None)
    return state


def get_train_step(use_grad_acc):

    @parallelize
    def train_step(state, batch, rng_key):
        def loss_fn(params):
            logits, new_model_state = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                batch["input_frames"],
                batch["attention_mask"],
                mutable=["batch_stats"],
                rngs={"dropout": rng_key})
            label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
            labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
            loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            loss = (label_mask * loss).sum() / label_mask.sum()
            # TODO(lmzheng): implement the correct rnn-t loss for transducer.
            return loss, new_model_state

        step = state.step
        dynamic_scale = state.dynamic_scale

        if dynamic_scale:
            # TOOD(lmzheng): handle gradient accumulation for this
            grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
            dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
            # dynamic loss takes care of averaging gradients across replicas
        else:
            if use_grad_acc:
                get_grad_fn = alpa.grad
            else:
                get_grad_fn = jax.grad

            grad_fn = get_grad_fn(loss_fn, has_aux=True)
            grads, aux = grad_fn(state.params)
        new_model_state = aux

        new_state = state.apply_gradients(
            grads=grads, batch_stats=new_model_state["batch_stats"])
        if dynamic_scale:
            # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
            # params should be restored (= skip this step).
            new_state = new_state.replace(
                opt_state=jax.tree_multimap(
                    functools.partial(jnp.where, is_fin),
                    new_state.opt_state,
                    state.opt_state),
                params=jax.tree_multimap(
                    functools.partial(jnp.where, is_fin),
                    new_state.params,
                    state.params))
            metrics["scale"] = dynamic_scale.scale

        return new_state

    return train_step


def benchmark_model_one_case(benchmark_case):
    print_used_time(None)

    # Model configs
    model_type = "conformer"
    batch_size, seq_len, input_dim, num_layers, conv_subsample_channel,\
        conv_kernel_size, hidden_size, num_heads, vocab_size,\
        mesh_dim0, mesh_dim1, num_micro_batches, force_data_parallel,\
        use_remat = benchmark_case

    dtype = jnp.float32

    # Parallel configs
    global_config.force_data_parallel = force_data_parallel

    if num_micro_batches > 1:
        use_grad_acc = True
        global_config.prefer_reduce_scatter = False
    else:
        use_grad_acc = False
        global_config.prefer_reduce_scatter = True
        num_micro_batches = None

    if args.local:
        physical_mesh = LocalPhysicalDeviceMesh(jax.devices())
    else:
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
    logical_mesh = physical_mesh.get_logical_mesh([mesh_dim0, mesh_dim1],
                                                  mesh_topology="tree",
                                                  inter_host_bandwidth=1,
                                                  intra_host_bandwidth=30)
    set_parallelize_options(devices=logical_mesh,
                            num_micro_batches=num_micro_batches)
    print_used_time("Setup device mesh")

    # Prepare input batch
    num_classes = 1000
    batch = {
        "input_frames": jnp.ones((batch_size, seq_len * 4, input_dim, 1), dtype=dtype),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
    }
    print_used_time("Prepare input")

    # Init train state
    if model_type == "conformer":
        model = ConformerForASRModule(ConformerConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            conv_subsample_channel=conv_subsample_channel,
            conv_kernel_size=conv_kernel_size,
        ), dtype=dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, batch)
    train_step = get_train_step(use_grad_acc)
    print_used_time("Create train state")
    param_count = compute_param_number(state.params)

    # Compile executable
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    physical_mesh.sync_workers()
    print_used_time("Compile (workers)")

    # Benchmark step time
    for i in range(args.niter):
        state = train_step(state, batch, rngkey)

    costs = executable.get_execution_time_costs(warmup=2)
    print_used_time("Benchmark")

    # Check sharding strategy
    objective = testing.last_compiled_auto_sharding_objective or 0.0
    alloc_mem = executable.get_total_allocation_size()
    hlo_text = executable.get_hlo_text()

    with open("last.hlo", "w") as fout:
        fout.write(hlo_text)
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all =\
        count_communication_primitives(hlo_text)
    print(f"#total: {n_total}, #all-reduce: {n_all_reduce}, "
          f"#all-gather: {n_all_gather}, #reduce-scatter: {n_reduce_scatter}, "
          f"#all-to-all: {n_all_to_all}")

    # Log benchmark results
    num_gpus = mesh_dim0 * mesh_dim1
    tflops = executable.flop_count / num_gpus / np.mean(costs) / 1e12
    heads = ["Model", "Model Config", "Parallel Config", "Param count",
             "Alloc Mem", "ILP Objective", "Mean Time", "Std Time", "TFLOPS"]
    values = [model_type, str(benchmark_case[:-5]), str(benchmark_case[-5:]),
              f"{param_count/1e9:.3f}", f"{alloc_mem/GB:.3f}", f"{objective:.2f}",
              f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}",
              f"{tflops:.2f}"]
    write_tsv(heads, values, f"result_{model_type}.tsv")

    physical_mesh.shutdown()


# B = batch_size, S = seq_len, I = input_dim, L = num_layers,
# C = conv_subsample_channel, K = conv_kernel_size,
# H = hidden_size, #head = num_heads, V = vocab_size,
# D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FD = force_data_parallel, CK = use_checkpoint

default_benchmark_suite = {  # key = number of gpus, value = a list of cases
1: [
    #B,   S,    I,   L, C,   K,  H,    #head      V   D0, D1, NB, FD,    CK
    (4,   1024, 128, 4, 256, 32, 2048, 2048//64,  32, 1,  1,  1,  False, False),
],

8: [
    #B,   S,    I,   L, C,   K,  H,    #head      V   D0, D1, NB, FD,    CK
    (8,   512,  128, 4, 512, 32, 4096, 4096//128, 32, 8,  1,  1,  False, False),
    #(8,   512,  128, 4, 512, 32, 4096, 4096//128, 32, 8,  1,  1,  True,  False),
],
}

oom_benchmark_suite = {  # key = number of gpus, value = a list of cases
}

benchmark_suites = {
    "default": default_benchmark_suite,
    "oom": oom_benchmark_suite,
}


def benchmark_all():
    if args.local:
        num_gpus = list_gpu_info().count("UUID")
    else:
        num_gpus = int(ray.cluster_resources()["GPU"])

    try:
        suite = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        suite = None

    if not suite:
        print(f"No available benchmark suite for {args.suite} on {num_gpus} GPUs")
        return

    for case in suite:
        # Backup global config
        backup = global_config.backup()

        benchmark_model_one_case(case)

        # Restore global config
        global_config.restore(backup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--niter", type=int, default=4,
        help="Number of benchmark iteration")
    parser.add_argument("--suite", choices=["default", "oom"], default="default",
        help="The benchmark suite")
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    args = parser.parse_args()

    if not args.local:
        ray.init(address="auto")
        jax.config.update("jax_platform_name", "cpu")

    global_config.use_dummy_value_for_benchmarking = True

    benchmark_all()

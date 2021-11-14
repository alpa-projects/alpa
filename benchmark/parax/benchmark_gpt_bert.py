import argparse
import os
import time
from functools import partial

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

import parax
from benchmark.util import compute_gpt_parameter_count
from parax import (parallelize, global_config, set_parallelize_options, testing,
                   DeviceCluster, PhysicalDeviceMesh, automatic_layer_slicing)
from parax.model.bert_model import BertConfig, FlaxBertForMaskedLMModule, TrainState
from parax.model.gpt_model import FlaxGPTForLMModule
from parax.util import (run_cmd, write_tsv, map_to_shape, list_gpu_info, benchmark_func,
                        count_communication_primitives, print_used_time)


GB = 1024 ** 3


def compute_data_parallel_cost(optimizer, logical_mesh, physical_mesh):
    """For debugging usage."""
    shapes = jax.tree_util.tree_map(lambda x : np.prod(x.shape), optimizer.target)
    sizes = jax.tree_util.tree_leaves(shapes)
    cost = 0
    print(logical_mesh.mesh_beta)
    for size in sizes:
        cost += logical_mesh.all_reduce_cost(size * 4, 0)
        #cost += physical_mesh.prof_result.estimate_all_reduce(((0,4), (1,5), (2,6), (3,7),), size / 4, "float32")
        #cost += physical_mesh.prof_result.estimate_all_reduce(((0,2,4,6,), (1,3,5,7)), size / 2, "float32")
        #cost += physical_mesh.prof_result.estimate_all_reduce(((0,1,2,3,4,5,6,7),), size, "float32")
    print(cost)


def compute_tflops(batch_size, seq_len, num_layers, hidden_size, vocab_size,
                   num_gpus, latency, checkpoint_activations=False):
    factor = 96 if checkpoint_activations else 72
    total_flop = factor * batch_size * seq_len * (hidden_size ** 2) * num_layers * \
          (1 + seq_len / (6 * hidden_size)) \
          + 6 * batch_size * seq_len * hidden_size * vocab_size
    # Note: if we use dot to compute forward embedding
    # then the last term in total_flops should be
    # "+ 10 * batch_size * seq_len * hidden_size * vocab_size".
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops


def load_profiling_result(physical_mesh):
    filename = physical_mesh.get_signature() + ".prof.pkl"
    if os.path.exists(filename):
        print(f"Load saved profiling results from {filename}")
        physical_mesh.load_profiling_result(filename)
        physical_mesh.prof_result.make_monotonic()
        physical_mesh.prof_result.multiply_scale(1e7)
    else:
        physical_mesh.profile_collective("all-reduce")
        print(f"Save profiling results to {filename}")
        physical_mesh.save_profiling_result(filename)


def create_train_state(rngkey, model, dtype, batch):
    params = model.init_dummy(rngkey, batch["input_ids"], batch["attention_mask"],
                              batch["token_type_ids"], batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax.chain(
        #optax.clip_by_global_norm(1.0),  # TODO(lmzheng): fix reduce-scatter for this
        optax.adamw(learning_rate=1e-2, mask=weight_decay_mask)
    )

    mixed_precision = (dtype == jnp.float16)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        mixed_precision=mixed_precision,
        dynamic_scale=None)
    return state
 

def get_train_step(grad_func, num_layers, dtype):

    @parallelize
    def train_step(state, batch, rng_key):
        def loss_func(params):
            rngs = {"dropout": rng_key}
            logits = state.apply_fn(params,
                                    batch["input_ids"],
                                    batch["attention_mask"],
                                    batch["token_type_ids"],
                                    batch["position_ids"],
                                    deterministic=True,
                                    rngs=rngs)[0]
            label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
            labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
            loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            loss = (label_mask * loss).sum() / label_mask.sum()
            return loss

        grads = grad_func(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        # TODO(lmzheng): add dynamic scale for mixed-precision training
        return new_state

    return train_step


def benchmark_gpt_bert_internal(physical_mesh, model_type, benchmark_case, niter):
    # Backup global config
    backup = global_config.backup()
    print_used_time(None)

    # Model configs
    batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,\
        mesh_dim0, mesh_dim1, num_micro_batches, force_data_parallel,\
        prefer_reduce_scatter, use_remat = benchmark_case
    dtype = jnp.float16

    # Parallel configs
    if num_micro_batches > 1:
        grad_func = parax.grad
    else:
        num_micro_batches = None
        grad_func = jax.grad

    global_config.force_data_parallel = force_data_parallel
    global_config.prefer_reduce_scatter = prefer_reduce_scatter

    logical_mesh = physical_mesh.get_logical_mesh([mesh_dim0, mesh_dim1],
                                                  mesh_topology="tree",
                                                  inter_host_bandwidth=1,
                                                  intra_host_bandwidth=30)
    set_parallelize_options(devices=logical_mesh, num_micro_batches=num_micro_batches)

    print_used_time("Setup device mesh")

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
    if model_type == "gpt":
        model = FlaxGPTForLMModule(BertConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            max_position_embeddings=seq_len,
            type_vocab_size=0,
            gradient_checkpointing=use_remat,
        ), dtype=dtype)
    elif model_type == "bert":
        model = FlaxBertForMaskedLMModule(BertConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            max_position_embeddings=seq_len,
            type_vocab_size=0,
            gradient_checkpointing=use_remat,
        ), dtype=dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, dtype, batch)
    print_used_time("Create train state")

    # Compile executable
    train_step = get_train_step(grad_func, num_layers, dtype)
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    physical_mesh.sync_workers()
    print_used_time("Compile (workers)")
    alloc_mem = executable.get_total_allocation_size()

    # Benchmark step time
    if alloc_mem > 30 * GB:
        # out of memory
        latencies = [-1]
    else:
        for i in range(niter):
            state = train_step(state, batch, rngkey)

        latencies = executable.get_execution_time_costs(warmup=2)
    print_used_time("Benchmark")

    # Check sharding strategy
    ilp_objective = testing.last_compiled_auto_sharding_objective or 0.0
    hlo_text = executable.get_hlo_text()

    with open("last.hlo", "w") as fout:
        fout.write(hlo_text)
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all =\
        count_communication_primitives(hlo_text)
    print(f"#total: {n_total}, #all-reduce: {n_all_reduce}, "
          f"#all-gather: {n_all_gather}, #reduce-scatter: {n_reduce_scatter}, "
          f"#all-to-all: {n_all_to_all}")

    # Compute statistics
    tflops = compute_tflops(batch_size, seq_len, num_layers,
                            hidden_size, vocab_size,
                            physical_mesh.total_devices,
                            np.mean(latencies))
    param_count = compute_gpt_parameter_count(num_layers, hidden_size, vocab_size)

    # Restore global config
    global_config.restore(backup)

    return latencies, alloc_mem, tflops, param_count, ilp_objective


def benchmark_one_case(case):
    # Launch physical mesh
    if args.local:
        physical_mesh = PhysicalDeviceMesh(jax.devices())
    else:
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()

    # Run benchmark
    result = benchmark_gpt_bert_internal(physical_mesh, args.model, case, args.niter)
    latencies, alloc_mem, tflops, param_count, ilp_objective = result

    # Log results
    heads = ["Model", "Model Config", "Parallel Config", "Param Count",
             "Alloc Mem", "ILP Objective", "Mean Latency", "Std Latency", "TFLOPS"]
    values = [args.model, case[:-6], case[-6:],
              f"{param_count/1e9:.3f}", f"{alloc_mem/GB:.3f}", f"{ilp_objective:.2f}",
              f"{np.mean(latencies):.3f}", f"{np.std(latencies):.3f}", f"{tflops:.2f}"]
    write_tsv(heads, values, f"result_{args.model}.tsv")

    physical_mesh.shutdown()


# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# #head = num_heads, D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FD = force_data_parallel,
# RS = prefer_reduce_scatter, CK = use_checkpoint

default_benchmark_suite = {  # key = number of gpus, value = a list of cases
1: [
    # B,  S,    H,    L,  #head,     V,     D0, D1, NB, FD,    RS,    CK
    (16,  512,  1024, 10, 1024//64,  25600, 1,  1,  1,  False, False, False),
    (8,  1024,  1536, 10, 1536//96,  25600, 1,  1,  1,  False, False, False),
],

4: [
    # B,   S,    H,    L,  #head,     V,     D0, D1, NB, FD,    RS,    CK
],

8: [
    # B,   S,    H,    L,  #head,     V,     D0, D1, NB, FD,    RS,    CK
    (256,  512,  1024, 10, 1024//64,  25600, 8,  1,  1,  False, True,  False),
    (512,  512,  1024, 10, 1024//64,  25600, 8,  1,  2,  False, True,  False),
    (8,    1024, 4096, 10, 4096//128, 25600, 8,  1,  1,  True,  True,  False),
    (8,    1024, 4096, 10, 4096//128, 25600, 2,  4,  1,  False, True,  False),
    (8,    1024, 4096, 10, 4096//128, 25600, 1,  8,  1,  False, True,  False),
    (8,    1024, 4096, 10, 4096//128, 25600, 1,  8,  1,  False, True,  True),
],

16: [
    # B,   S,    H,    L,  #head,     V,     D0, D1, NB, FD,    RS,    CK
    (512,  512,  1024, 10, 1024//64,  25600, 16, 1,  1,  False, True,  False),
    (2048, 512,  1024, 10, 1024//64,  25600, 16, 1,  4,  False, True,  False),
    (16,   1024, 4096, 10, 4096//128, 25600, 2,  8,  1,  False, True,  False),
    (64,   1024, 4096, 10, 4096//128, 25600, 2,  8,  4,  False, True,  False),
    (16,   1024, 4096, 10, 4096//128, 25600, 16, 1,  1,  False, True,  False),
    #(64,   1024, 4096, 10, 4096//128, 25600, 16, 1,  4,  False, True,  False),
]
}


benchmark_suites = {
    "default": default_benchmark_suite,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--niter", type=int, default=10,
        help="Number of benchmark iteration")
    parser.add_argument("--suite", choices=["default"], default="default")
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    args = parser.parse_args()

    # Set global environments
    if args.local:
        num_gpus = list_gpu_info().count("UUID")
    else:
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

import argparse
import copy
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


def compute_parameter_count(num_layers, hidden_size, vocab_size):
    return num_layers * (
            # self-attention
            hidden_size * (3 * hidden_size + 1) + 
            hidden_size * (hidden_size + 1) + 
            # mlp
            hidden_size * (4 * hidden_size + 1) +
            hidden_size * 4 * (hidden_size + 1) +
            # layer norm
            hidden_size * 4
           ) + vocab_size * (hidden_size + 1)


def load_profiling_result():
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


def create_train_state(rngkey, model, batch):
    params = model.init_dummy(rngkey, batch["input_ids"], batch["attention_mask"],
                              batch["token_type_ids"], batch["position_ids"])
    tx = optax.adam(learning_rate=1e-2)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dynamic_scale=None)
    return state
 

def get_train_step(grad_func, num_layers, use_remat, dtype):

    @parallelize
    def train_step(state, batch, rng_key):
        @partial(automatic_layer_slicing, layer_num=num_layers, use_remat=use_remat)
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

        params = jax.tree_util.tree_map(lambda x : jnp.asarray(x, dtype), state.params)
        grads = grad_func(loss_func)(params)
        new_state = state.apply_gradients(grads=grads)
        # TODO(lmzheng): add dynamic scale for mixed-precision training
        return new_state

    return train_step


def benchmark_model_one_case(benchmark_case):
    print_used_time(None)

    # Model configs
    model_type = args.model
    batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,\
        mesh_dim0, mesh_dim1, num_micro_batches, force_data_parallel,\
        use_remat = benchmark_case
    dtype = jnp.float16

    # Parallel configs
    global_config.force_data_parallel = force_data_parallel

    if num_micro_batches > 1:
        grad_func = parax.grad
        global_config.prefer_reduce_scatter = False
    else:
        num_micro_batches = None
        grad_func = jax.grad
        global_config.prefer_reduce_scatter = True

    if args.local:
        physical_mesh = PhysicalDeviceMesh(jax.devices())
    else:
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
    logical_mesh = physical_mesh.get_logical_mesh([mesh_dim0, mesh_dim1],
                                                  mesh_topology="tree",
                                                  inter_host_bandwidth=1,
                                                  intra_host_bandwidth=30)
    set_parallelize_options(devices=logical_mesh, num_micro_batches=num_micro_batches)

    # Load profiling results
    if args.use_profiling:
        load_profiling_result(physical_mesh)
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
        ), dtype=dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, batch)
    print_used_time("Create train state")

    # Compile executable
    train_step = get_train_step(grad_func, num_layers, use_remat, dtype)
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
    real_mem = executable.get_total_allocation_size()
    hlo_text = executable.get_hlo_text()

    with open("last.hlo", "w") as fout:
        fout.write(hlo_text)
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ =\
        count_communication_primitives(hlo_text)
    print(f"#total: {n_total}, #all-reduce: {n_all_reduce}, "
          f"#all-gather: {n_all_gather}, #reduce-scatter: {n_reduce_scatter}")

    # Log benchmark results
    tflops = compute_tflops(batch_size, seq_len, num_layers,
                            hidden_size, vocab_size,
                            physical_mesh.total_devices,
                            np.mean(costs))
    parameter_count = compute_parameter_count(num_layers, hidden_size, vocab_size)
    heads = ["Type", "Model Config", "Parallel Config", "Parameter Count",
             "Peak Mem", "Objective", "Mean Time", "Std Time", "TFLOPS"]
    values = [model_type, str(benchmark_case[:-5]), str(benchmark_case[-5:]),
              f"{parameter_count/1e9:.3f}", f"{real_mem/GB:.3f}", f"{objective:.2f}",
              f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}", f"{tflops:.2f}"]
    write_tsv(heads, values, f"result_{model_type}.tsv")

    physical_mesh.shutdown()


# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# #head = num_heads, D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FD = force_data_parallel, CK = use_checkpoint

default_benchmark_suite = {  # key = number of gpus, value = a list of cases
1: [
    # B,  S,    H,    L,  #head,     V,     D0, D1, NB, FD,    CK
    (16,  512,  1024, 10, 1024//64,  25600, 1,  1,  1,  False, False),
    (8,   1024, 1536, 10, 1536//96,  25600, 1,  1,  1,  False, False),
],

4: [
    # B,  S,    H,    L,  #head,     V,     D0, D1, NB, FD,    CK
],

8: [
    # B,  S,    H,    L,  #head,     V,     D0, D1, NB, FD,    CK
    (256, 512,  1024, 10, 1024//64,  25600, 8,  1,  1,  False, False),
    (8,   1024, 4096, 10, 4096//128, 25600, 8,  1,  1,  True,  False),
    (8,   1024, 4096, 10, 4096//128, 25600, 2,  4,  1,  False, False),
    (8,   1024, 4096, 10, 4096//128, 25600, 1,  8,  1,  False, False),
],

16: [
    # B,   S,    H,    L,  #head,     V,     D0, D1, NB, FD,    CK
    (512,  512,  1024, 10, 1024//64,  25600, 16, 1,  1,  False, False),
    (2048, 512,  1024, 10, 1024//64,  25600, 16, 1,  4,  False, False),
    (16,   1024, 4096, 10, 4096//128, 25600, 2,  8,  1,  False, False),
    (64,   1024, 4096, 10, 4096//128, 25600, 2,  8,  4,  False, False),
]
}


benchmark_suites = {
    "default": default_benchmark_suite,
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
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--niter", type=int, default=10,
        help="Number of benchmark iteration")
    parser.add_argument("--suite", choices=["default"], default="default")
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    args = parser.parse_args()

    if not args.local:
        ray.init(address="auto")
        jax.config.update('jax_platform_name', 'cpu')

    global_config.use_dummy_value_for_benchmarking = True

    benchmark_all()

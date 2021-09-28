import argparse
import time

import copy
import jax
import jax.numpy as jnp
import numpy as np
import os
import parax
import ray
from flax import optim

from parax import (parallelize, global_config, set_parallelize_options, testing,
                   DeviceCluster, PhysicalDeviceMesh, mark_pipeline, manual_layer_slicing)
from parax.model.bert_model import BertConfig, FlaxBertForMaskedLMModule
from parax.model.gpt_model import FlaxGPTForLMModule
from parax.util import (write_tsv, list_gpu_info, benchmark_func,
                        count_communication_primitives)
from parax.pipeline_parallel.runtime import timer_names

GB = 1024 ** 3


tic = time.time()
def log_time_stamp(message):
    global tic
    if message:
        print(f" - {message}: {time.time() - tic:.2f} s")
    tic = time.time()


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


def benchmark_transformer_one_case(benchmark_case, use_profiling):
    log_time_stamp(None)

    # Model configs
    model_type = args.model
    batch_size, seq_len, hidden_size, num_layers, num_heads, \
    vocab_size, mesh_dim0, mesh_dim1, pipeline_mp_size, num_micro_batches, \
    force_data_parallel = benchmark_case
    dtype = jnp.float16

    # Parallel configs
    global_config.force_data_parallel = force_data_parallel
    global_config.prefer_reduce_scatter = False

    device_cluster = DeviceCluster()
    virtual_mesh = device_cluster.get_virtual_mesh()
    # logical_mesh = physical_mesh.get_logical_mesh([mesh_dim0, mesh_dim1],
    #                                               mesh_topology="tree",
    #                                               inter_host_bandwidth=1,
    #                                               intra_host_bandwidth=30)
    set_parallelize_options(devices=virtual_mesh, strategy="3d_parallel")


    @parallelize(donate_argnums=())
    def train_step(optimizer, batch, rng_key, apply_func):

        def loss_func(params):
            rngs = {"dropout": rng_key}
            if pipeline_mp_size > 1:
                mark_pipeline(name="0", mark_type="start")

            logits = apply_func(params,
                                batch["input_ids"],
                                batch["attention_mask"],
                                batch["token_type_ids"],
                                batch["position_ids"],
                                deterministic=True,
                                rngs=rngs)[0]
            label_mask = jnp.where(batch["labels"]  > 0, 1.0, 0.0)
            labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
            loss = - jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            loss = (label_mask * loss).sum() / label_mask.sum()
            # TODO(lmzheng): add dynamic scale for mixed-precision training
            if pipeline_mp_size > 1:
                mark_pipeline(name=str(pipeline_mp_size - 1), mark_type="end")

            return loss

        # # pipeline marker
        # if pipeline_mp_size > 1:
        #     mark_pipeline(optimizer.target, name="-2", mark_type="start")
        # params = jax.tree_util.tree_map(lambda x : jnp.asarray(x, dtype), optimizer.target)
        # grad = jax.grad(loss_func)(params)
        # new_optimizer = optimizer.apply_gradient(grad)
        if pipeline_mp_size > 1:
            loss_func = manual_layer_slicing(loss_func)

        grad = jax.grad(loss_func, argnums=(0))(optimizer.target)
        # new_optimizer = optimizer.apply_gradient(grad)
        # return new_optimizer
        return grad

    # Prepare input batch
    tmp_dtype = jnp.int32
    batch = {
        "input_ids": jnp.ones((batch_size, seq_len), dtype=tmp_dtype),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=tmp_dtype),
        "token_type_ids": jnp.ones((batch_size, seq_len), dtype=tmp_dtype),
        "position_ids": jnp.ones((batch_size, seq_len), dtype=tmp_dtype),
        "labels": jnp.ones((batch_size, seq_len), dtype=tmp_dtype),
    }
    log_time_stamp("Prepare input")

    # Init model and optimizer
    if model_type == "gpt":
        model = FlaxGPTForLMModule(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
        ), dtype=dtype)
    elif model_type == "bert":
        model = FlaxBertForMaskedLMModule(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
        ), dtype=dtype)
    elif model_type == "bert_pipeline":
        model = FlaxBertForMaskedLMModule(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
            pipeline_mp_size=pipeline_mp_size
        ), dtype=dtype)
    elif model_type == "gpt_pipeline":
        model = FlaxGPTForLMModule(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
            pipeline_mp_size=pipeline_mp_size
        ), dtype=dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    params = model.init_dummy(rngkey, batch["input_ids"], batch["attention_mask"],
                              batch["token_type_ids"], batch["position_ids"])
    optimizer = optim.Adam(1e-2).create(params)
    del params
    log_time_stamp("Init model and optimizer")

    # Compile executable
    executable = train_step.get_executable(optimizer, batch, rngkey, model.apply)
    log_time_stamp("Compile (driver)")

    for i in range(args.niter):
        train_step(optimizer, batch, rngkey, model.apply)

    # def run_func():
    #     nonlocal optimizer
    #     train_step(optimizer, batch, rngkey, model.apply)

    timer_name = "process_output"
    costs = executable.get_execution_time_costs(timer_name=timer_name)
    real_mem = -1
    objective = -1

    # Log benchmark results
    # tflops = compute_tflops(batch_size, seq_len, num_layers,
    #                         hidden_size, vocab_size,
    #                         physical_mesh.total_devices,
    #                         np.mean(costs))
    # parameter_count = compute_parameter_count(num_layers, hidden_size, vocab_size)
    heads = ["Type", "Case", "Mesh Shape", "Peak Mem", "Objective", "Mean Time", "Std Time"]
    values = ["transformer-layer", str(benchmark_case[:-5]), str(benchmark_case[-5:]),
              f"{real_mem/GB:.3f}", f"{objective:.2f}",
              f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}"]
    write_tsv(heads, values, f"result_{model_type}.tsv")

    # physical_mesh.shutdown()


# B = global_batch_size, S = seq_len,
# H = hidden_size, L = num_layers, V = vocab_size, #head = num_heads,
# DP = data_parallel, TP = tensor_model_parallel, PP = pipeline_model_parallel,
# NB = num_micro_batches
# DI = ddp_implementation, CK = checkpoint_activations
# FD = force data-parallel

# w/ pipeliening
benchmark_suite_1_gpu = [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, FD
    (16,  512,  1024, 24, 1024//64,  32000, 1,  1,  1,  1,  False),
    (8,   1024, 1536, 16, 1536//96,  32000, 1,  1,  1,  1,  False),
]

benchmark_suite_4_gpu = [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, FD

    (2,  512,  1024, 24, 1024//64,  32000, 1,  1,  2,  1,  False),
    # (16,  512,  1024, 24, 1024//64,  32000, 1,  1,  2,  1,  False),
    # (8,   1024, 1536, 16, 1536//96,  32000, 1,  1,  2,  1,  False),
]

benchmark_suite_8_gpu = [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, FD
    (128, 512,  1024, 24, 1024//64,  32000, 8,  1,  1,  1,  False),
    (256, 512,  1024, 24, 1024//64,  32000, 8,  1,  1,  2,  False),
    (8,   1024, 4096, 20, 4096//128, 32000, 1,  8,  1,  1,  False),
    (16,  1024, 4096, 20, 4096//128, 32000, 1,  8,  1,  2,  False),
    (256, 1024, 4096, 20, 4096//128, 32000, 1,  8,  1,  32, False),
]

benchmark_suite_16_gpu = [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, FD
    (256, 512,  1024, 24, 1024//64,  32000, 16, 1,  1,  1,  False),
    (512, 512,  1024, 24, 1024//64,  32000, 16, 1,  1,  2,  False),
    (16,  1024, 4096, 20, 4096//128, 32000, 2,  8,  1,  1,  False),
    (256, 1024, 4096, 20, 4096//128, 32000, 2,  8,  1,  16, False),
]

def benchmark_all(use_profiling):
    num_gpus = ray.cluster_resources()["GPU"]
    benchmark_suites = {
        # 1: benchmark_suite_1_gpu,
        4: benchmark_suite_4_gpu,
        # 8: benchmark_suite_8_gpu,
        # 16: benchmark_suite_16_gpu,
    }

    for case in benchmark_suites[int(num_gpus)]:
        # Backup global config
        old_global_config = copy.deepcopy(global_config.__dict__)

        benchmark_transformer_one_case(case, use_profiling)

        # Restore global config
        global_config.__dict__ = old_global_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--niter", type=int, default=10)
    args = parser.parse_args()

    ray.init(address="auto")
    jax.config.update('jax_platform_name', 'cpu')

    global_config.use_dummy_value_for_benchmarking = True

    benchmark_all(args.use_profiling)

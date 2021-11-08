import argparse

import jax
import jax.numpy as jnp
import numpy as np
import ray
import optax

import parax
from benchmark.parax.benchmark_transformer_layer_3d import report_pipeline_breakdown
from parax import (parallelize, global_config, set_parallelize_options,
                   DeviceCluster, mark_pipeline, manual_layer_slicing)
from parax.model.bert_model import BertConfig, FlaxBertForMaskedLMModule, TrainState
from parax.model.gpt_model import FlaxGPTForLMModule
from parax.util import write_tsv, print_used_time
from benchmark.parax.benchmark_gpt_bert import compute_parameter_count, compute_tflops

GB = 1024 ** 3


def create_train_state(rngkey, model, batch, dtype):
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


def get_train_step(grad_func, num_layers, use_remat, pipeline_mp_size, dtype):

    @parallelize
    def train_step(state, batch, rng_key):

        # @partial(automatic_layer_slicing, layer_num=num_layers, use_remat=use_remat)
        def loss_func(params):
            rngs = {"dropout": rng_key}
            if pipeline_mp_size > 1:
                mark_pipeline(name="0", mark_type="start")
            logits = state.apply_fn(params,
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
            if pipeline_mp_size > 1:
                mark_pipeline(name=str(pipeline_mp_size - 1), mark_type="end")
            return loss

        if pipeline_mp_size > 1:
            loss_func = manual_layer_slicing(loss_func)
        # params = jax.tree_util.tree_map(lambda x: x, state.params)
        grads = grad_func(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state

    return train_step

def benchmark_one_case(benchmark_case):
    backup = global_config.backup()
    print_used_time(None)

    # Model configs
    model_type = args.model

    batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size, \
    mesh_dim0, mesh_dim1, pipeline_mp_size, num_micro_batches, force_data_parallel, \
    prefer_reduce_scatter = benchmark_case
    dtype = jnp.float16

    # Parallel configs
    grad_func = parax.grad

    global_config.force_data_parallel = force_data_parallel
    global_config.prefer_reduce_scatter = prefer_reduce_scatter

    device_cluster = DeviceCluster()
    virtual_mesh = device_cluster.get_virtual_mesh()
    # logical_mesh = physical_mesh.get_logical_mesh([mesh_dim0, mesh_dim1],
    #                                               mesh_topology="tree",
    #                                               inter_host_bandwidth=1,
    #                                               intra_host_bandwidth=30)
    set_parallelize_options(devices=virtual_mesh,
                            strategy="3d_parallel",
                            num_micro_batches=num_micro_batches)
    print_used_time("Setup device mesh")


    # Prepare input batch
    # Note: there will be an input conversion.
    input_dtype = jnp.int32
    batch = {
        "input_ids": jnp.ones((batch_size, seq_len), dtype=input_dtype),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=input_dtype),
        "token_type_ids": jnp.ones((batch_size, seq_len), dtype=input_dtype),
        "position_ids": jnp.ones((batch_size, seq_len), dtype=input_dtype),
        "labels": jnp.ones((batch_size, seq_len), dtype=input_dtype),
    }
    print_used_time("Prepare input")

    if model_type == "bert":
        model = FlaxBertForMaskedLMModule(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
            pipeline_mp_size=pipeline_mp_size
        ), dtype=dtype)
    elif model_type == "gpt":
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
    state = create_train_state(rngkey, model, batch, dtype)
    print_used_time("Create train state")

    # compile executable
    train_step = get_train_step(grad_func, num_layers, False, pipeline_mp_size, jnp.float16)
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    for i in range(args.niter):
        state = train_step(state, batch, rngkey)

    timer_name = "overall"
    overall_costs = executable.get_execution_time_costs(warmup=0, timer_name=timer_name)
    print_used_time("Benchmark")

    # Compute statistics
    tflops = compute_tflops(batch_size, seq_len, num_layers,
                            hidden_size, vocab_size,
                            virtual_mesh.total_devices,
                            np.mean(overall_costs[2:]))
    parameter_count = compute_parameter_count(num_layers, hidden_size, vocab_size)


    report_pipeline_breakdown(executable, ["resharding_send", "resharding_recv", "compute"], args.niter)
    heads = ["Type", "Model Config", "Parallel Config",  "# Microbatch", "Mean Time",
             "Std Time", "#Params", "TFLOPs"]
    values = [model_type, str(benchmark_case[:-6]), str(benchmark_case[-6:-1]), f"{benchmark_case[-3]:.3f}",
              f"{np.mean(overall_costs[2:]):.3f}", f"{np.std(overall_costs[2:]):.3f}", str(parameter_count),
              str(tflops)]
    write_tsv(heads, values, f"result_{model_type}.tsv")

    executable.shutdown()


# B = global_batch_size, S = seq_len,
# H = hidden_size, L = num_layers, V = vocab_size, #head = num_heads,
# DP = data_parallel, TP = tensor_model_parallel, PP = pipeline_model_parallel,
# NB = num_micro_batches, FD = force data-parallel
# RS = prefer_reduce_scatter

# w/ pipeliening
default_benchmark_suite = {

8: [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, FD, RS
    # GPT-2 355M, DP + PP2
    (16,  1024,  1024, 24, 1024//64,  51200, 4,  1,  2,  1,  True, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 4,  1,  2,  1,  True, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 4,  1,  2,  2,  True, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 4,  1,  2,  4,  True, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 4,  1,  2,  8,  True, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 4,  1,  2,  16,  True, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 4,  1,  2,  32,  True, False),

    # GPT-3 355M, auto sharding + PP2
    (32,  1024,  1024, 24, 1024//64,  51200, 4,  1,  2,  1,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 4,  1,  2,  2,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 4,  1,  2,  4,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 4,  1,  2,  8,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 4,  1,  2,  16,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 4,  1,  2,  32,  False, False),

    # GPT-3 355M, DP + PP4
    (32,  1024,  1024, 24, 1024//64,  51200, 2,  1,  4,  1,  True, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 2,  1,  4,  2,  True, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 2,  1,  4,  4,  True, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 2,  1,  4,  8,  True, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 2,  1,  4,  16,  True, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 2,  1,  4,  32,  True, False),

    # GPT-3 355M, auto sharding + PP4
    (32,  1024,  1024, 24, 1024//64,  51200, 2,  1,  4,  1,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 2,  1,  4,  2,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 2,  1,  4,  4,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 2,  1,  4,  8,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 2,  1,  4,  16,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 2,  1,  4,  32,  False, False),

    # GPT-3 355M, PP8
    # (16,  1024,  1024, 24, 1024//64,  51200, 2,  2,  2,  8,  False, False),  # sanity check case
    (32,  1024,  1024, 24, 1024//64,  51200, 4,  2,  2,  32,  True, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 1,  1,  8,  2,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 1,  1,  8,  4,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 1,  1,  8,  8,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 1,  1,  8,  16,  False, False),
    (32,  1024,  1024, 24, 1024//64,  51200, 1,  1,  8,  32,  False, False),
],

16: [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, FD, RS
]
}

benchmark_suites = {
    "default": default_benchmark_suite
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--niter", type=int, default=10)
    parser.add_argument("--suite", choices=["default"], default="default")
    args = parser.parse_args()

    ray.init(address="auto")
    jax.config.update('jax_platform_name', 'cpu')
    num_gpus = int(ray.cluster_resources()["GPU"])

    global_config.use_dummy_value_for_benchmarking = True

    try:
        suite = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        suite = None

    if not suite:
        print(f"No available benchmark suite for {args.suite} on {num_gpus} GPUs")
        exit()

    for case in suite:
        benchmark_one_case(case)

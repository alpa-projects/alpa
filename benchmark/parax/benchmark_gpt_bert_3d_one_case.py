import argparse

import jax
import jax.numpy as jnp
import numpy as np
import ray
import optax

import parax
from benchmark.parax.benchmark_transformer_layer_3d import report_pipeline_breakdown
from benchmark.util import compute_gpt_parameter_count, compute_gpt_tflops
from parax import (parallelize, global_config, set_parallelize_options,
                   DeviceCluster, mark_pipeline, manual_layer_slicing, automatic_layer_slicing)
from parax.model.bert_model import BertConfig, FlaxBertForMaskedLMModule
from parax.model.model_util import TrainState
from parax.model.gpt_model import FlaxGPTForLMModule
from parax.util import write_tsv, print_used_time

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


def get_train_step(grad_func, num_layers, use_remat, pipeline_mp_size, dtype, auto_layer=False):

    add_pipeline_marker = ((not auto_layer) and (pipeline_mp_size > 1))

    @parallelize
    def train_step(state, batch, rng_key):

        # @partial(automatic_layer_slicing, layer_num=num_layers, use_remat=use_remat)
        def loss_func(params):
            rngs = {"dropout": rng_key}
            if add_pipeline_marker:
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
            if add_pipeline_marker:
                mark_pipeline(name=str(pipeline_mp_size - 1), mark_type="end")
            return loss

        if add_pipeline_marker:
            loss_func = manual_layer_slicing(loss_func)
        elif auto_layer:
            loss_func = automatic_layer_slicing(loss_func, pipeline_mp_size, use_pipeline=True)
        grads = grad_func(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        # TODO(lmzheng): add dynamic scaling for mixed-precision training
        return new_state

    return train_step

def benchmark_one_case(benchmark_case, external_args):
    backup = global_config.backup()
    print_used_time(None)

    # Model configs
    model_type = external_args.model

    (batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,
     l_dim0, l_dim1, p_dim0, p_dim1, pipeline_mp_size, num_micro_batches, force_data_parallel,
     use_remat, auto_layer, auto_stage) = benchmark_case

    dtype = jnp.float16
    tie_word_embeddings = False

    # Parallel configs
    grad_func = parax.grad

    global_config.force_data_parallel = force_data_parallel
    global_config.prefer_reduce_scatter = False

    device_cluster = DeviceCluster()
    virtual_mesh = device_cluster.get_virtual_mesh()
    if not auto_stage:
        set_parallelize_options(devices=virtual_mesh,
                                strategy="3d_parallel",
                                num_micro_batches=num_micro_batches,
                                sub_physical_mesh_shapes=[(p_dim0, p_dim1)] * pipeline_mp_size,
                                sub_logical_mesh_shapes=[(l_dim0, l_dim1)] * pipeline_mp_size)
    else:
        set_parallelize_options(devices=virtual_mesh,
                                strategy="3d_parallel",
                                pipeline_stage_mode="auto_gpipe",
                                num_micro_batches=num_micro_batches)

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

    add_manual_layer_slicing_marker = ((not auto_layer) and (pipeline_mp_size > 1))

    if model_type == "bert":
        model = FlaxBertForMaskedLMModule(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
            pipeline_mp_size=pipeline_mp_size,
            gradient_checkpointing=use_remat,
            tie_word_embeddings=tie_word_embeddings,
            add_manual_pipeline_markers=add_manual_layer_slicing_marker,
        ), dtype=dtype)
    elif model_type == "gpt":
        model = FlaxGPTForLMModule(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
            pipeline_mp_size=pipeline_mp_size,
            gradient_checkpointing=use_remat,
            tie_word_embeddings=tie_word_embeddings,
            add_manual_pipeline_markers=add_manual_layer_slicing_marker,
        ), dtype=dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, batch, dtype)
    print_used_time("Create train state")

    # compile executable
    train_step = get_train_step(grad_func, num_layers, False, pipeline_mp_size, jnp.float16, auto_layer)
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    executable.sync()
    print_used_time("Compile (worker)")

    for i in range(external_args.niter):
        state = train_step(state, batch, rngkey)

    timer_name = "overall"
    overall_costs = executable.get_execution_time_costs(warmup=0, timer_name=timer_name)
    print_used_time("Benchmark")

    # Compute statistics
    tflops = compute_gpt_tflops(batch_size, seq_len, num_layers,
                                hidden_size, vocab_size,
                                virtual_mesh.total_devices,
                                np.mean(overall_costs[2:]))
    tflops_ckpt = compute_gpt_tflops(batch_size, seq_len, num_layers,
                                     hidden_size, vocab_size,
                                     virtual_mesh.total_devices,
                                     np.mean(overall_costs[2:]), True)
    parameter_count = compute_gpt_parameter_count(num_layers, hidden_size, vocab_size)

    # report_pipeline_breakdown(executable, ["resharding_send", "resharding_recv", "compute"], external_args.niter)
    heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape", "#Microbatch",
             "Force DP", "Remat", "Mean Time", "Std Time", "#Params", "TFLOPs", "TFLOPs (ckpt)"]
    paralell_config = (l_dim0, l_dim1, pipeline_mp_size)
    values = [model_type, str(benchmark_case[:5]), str(paralell_config), str(benchmark_case[8:10]),
              str(benchmark_case[11]), str(benchmark_case[12]), str(benchmark_case[13]),
              f"{np.mean(overall_costs[2:]):.3f}", f"{np.std(overall_costs[2:]):.3f}",
              f"{parameter_count/1e9:.3f}", f"{tflops:.2f}", f"{tflops_ckpt:.2f}"]
    write_tsv(heads, values, f"{model_type}_parax_{external_args.exp_name}.tsv")

    executable.shutdown()


def setup_benchmark():
    jax.config.update('jax_platform_name', 'cpu')
    global_config.use_dummy_value_for_benchmarking = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--niter", type=int, default=7)
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="result")
    args = parser.parse_args()
    case = eval(args.case)
    setup_benchmark()
    ray.init(address="auto")
    benchmark_one_case(case, args)
    ray.shutdown()

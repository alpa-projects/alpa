import argparse
import pickle

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
from parax.util import print_used_time, run_cmd

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

    add_pipeline_marker = ((not auto_layer) and (pipeline_mp_size >= 1))

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

def benchmark_gpt_bert_internal(model_type, benchmark_case, niter):
    backup = global_config.backup()
    print_used_time(None)

    # Model configs
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
    virtual_mesh = device_cluster.get_virtual_physical_mesh()
    if not auto_stage:
        set_parallelize_options(devices=virtual_mesh,
                                strategy="3d_parallel",
                                num_micro_batches=num_micro_batches,
                                sub_physical_mesh_shapes=[(p_dim0, p_dim1)] * pipeline_mp_size,
                                sub_logical_mesh_shapes=[(l_dim0, l_dim1)] * pipeline_mp_size,
                                pipeline_parallel_schedule="1f1b")
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

    # dump hlo ir for debugging
    stage_hlo_texts = executable.get_hlo_text()
    for i in range(len(stage_hlo_texts)):
        with open(f"tmp/stage_{i}.hlo", "w") as fout:
            fout.write(stage_hlo_texts[i])
    with open(f"tmp/resharding_tasks.txt", "w") as fout:
        fout.write(executable.print_resharding_tasks())

    executable.sync()
    print_used_time("Compile (worker)")

    for i in range(niter):
        state = train_step(state, batch, rngkey)

    timer_name = "overall"
    latencies = executable.get_execution_time_costs(warmup=0, timer_name=timer_name)[2:]
    print_used_time("Benchmark")

    mem_allocated = executable.get_memory_allocated()
    max_mem_allocated = executable.get_max_memory_allocated()

    # Compute statistics
    tflops = compute_gpt_tflops(batch_size, seq_len, num_layers,
                                hidden_size, vocab_size,
                                virtual_mesh.total_devices,
                                np.mean(latencies))
    tflops_ckpt = compute_gpt_tflops(batch_size, seq_len, num_layers,
                                     hidden_size, vocab_size,
                                     virtual_mesh.total_devices,
                                     np.mean(latencies), True)
    parameter_count = compute_gpt_parameter_count(num_layers, hidden_size, vocab_size)
    # report_pipeline_breakdown(executable, ["resharding_send", "resharding_recv", "compute", "alloc"], niter)
    executable.shutdown()
    return parameter_count, mem_allocated, max_mem_allocated, latencies, tflops, tflops_ckpt


TMP_PICKLE_FILE_NAME = "tmp/tmp_transfer.pkl"


def benchmark_one_case(model, case, niter, use_separate_process=False, dump_result=False):
    if not use_separate_process:
        ray.init(address="auto", ignore_reinit_error=True)
        jax.config.update('jax_platform_name', 'cpu')
        global_config.use_dummy_value_for_benchmarking = True

        result = benchmark_gpt_bert_internal(model, case, niter)
    else:
        # Launch a new process for benchmark to isolate errors.
        # Get the return data via pickle.
        run_cmd(f"rm -rf {TMP_PICKLE_FILE_NAME}")
        ret = run_cmd("python3 benchmark_gpt_bert_3d_one_case.py "
                     f"--model {model} "
                     f"--niter {niter} "
                     f'--case "{case}" '
                     f"--dump-result ")
        if ret == 0:
            result = pickle.load(open(TMP_PICKLE_FILE_NAME, "rb"))
        else:
            result = -1, -1, -1, [-1], -1, -1

    if dump_result:
        pickle.dump(result, open(TMP_PICKLE_FILE_NAME, "wb"))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--niter", type=int, default=7)
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--dump-result", action="store_true",
        help="Dump results into a temporary pickle file")
    args = parser.parse_args()

    run_cmd("mkdir -p tmp")
    case = eval(args.case)
    benchmark_one_case(args.model, case, args.niter,
                       use_separate_process=False, dump_result=args.dump_result)

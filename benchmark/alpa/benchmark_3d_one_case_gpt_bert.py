"""Benchmark one case of inter-op + intra-op parallelism."""
import jax
import jax.numpy as jnp
import numpy as np
import optax

import alpa
from alpa import (parallelize, get_global_cluster,
                  set_global_virtual_physical_mesh, automatic_remat,
                  global_config)
from alpa.model.bert_model import BertConfig, FlaxBertForMaskedLMModule
from alpa.model.model_util import TrainState
from alpa.model.gpt_model import FlaxGPTForLMModule
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.util import print_used_time

from benchmark.util import compute_gpt_parameter_count, compute_gpt_tflops
from parallel_option import (get_pipeshard_parallel_method,
                             get_shard_parallel_method,
                             compile_and_benchmark_pipeshard_executable,
                             compile_and_benchmark_shard_executable)


def report_pipeline_breakdown(executable, timer_names, niter):
    overall_costs = executable.get_execution_time_costs(timer_name="overall")

    print(">>> overall: {}...".format(overall_costs))
    other_percentage = [100.0] * niter
    other = overall_costs
    for timer_name in timer_names:
        costs = executable.get_execution_time_costs(timer_name=timer_name)
        if len(costs) == 0:
            costs = [0.0] * niter
        percentage = [
            cost / overall_costs[i] * 100 for i, cost in enumerate(costs)
        ]
        other = [remain - costs[i] for i, remain in enumerate(other)]
        other_percentage = [
            remain - percentage[i] for i, remain in enumerate(other_percentage)
        ]
        strs = []
        for i, cost in enumerate(costs):
            strs.append(str(cost) + f" ({percentage[i]:.1f}) ")
        print_string = ",".join(strs)
        print(">>> {}: {}".format(timer_name, print_string))

    # print unknown overhead
    strs = []
    for i, remain in enumerate(other):
        strs.append(" " + str(remain) + f" ({other_percentage[i]:.1f})")
    print_string = ",".join(strs)
    print(">>> {}: {}".format("Others: ", print_string))


def create_train_state(rngkey, model, batch, dtype):
    params = model.init_dummy(rngkey, batch["input_ids"],
                              batch["attention_mask"], batch["token_type_ids"],
                              batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax.chain(
        #optax.clip_by_global_norm(1.0),  # TODO(lmzheng): fix reduce-scatter for this
        optax.adamw(learning_rate=1e-2, mask=weight_decay_mask))
    mixed_precision = (dtype == jnp.float16)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              mixed_precision=mixed_precision,
                              dynamic_scale=None)
    return state


def create_train_state_aval(rngkey, model, batch, dtype):
    params = jax.eval_shape(model.init, rngkey, batch["input_ids"],
                            batch["attention_mask"], batch["token_type_ids"],
                            batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax.chain(
        #optax.clip_by_global_norm(1.0),  # TODO(lmzheng): fix reduce-scatter for this
        optax.adamw(learning_rate=1e-2, mask=weight_decay_mask))
    mixed_precision = (dtype == jnp.float16)
    state = TrainState.create_aval(apply_fn=model.apply,
                                   params=params,
                                   tx=tx,
                                   mixed_precision=mixed_precision,
                                   dynamic_scale=None)
    return state


def get_train_step(parallel_method,
                   use_fine_grained_remat=False,
                   fine_grained_remat_num_layers=None,
                   grad_func=None):

    if grad_func is None:
        grad_func = alpa.grad

    @parallelize(method=parallel_method)
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
            loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1),
                            axis=-1)
            loss = (label_mask * loss).sum() / label_mask.sum()
            return loss

        if use_fine_grained_remat:
            loss_func = automatic_remat(loss_func,
                                        layer_num=fine_grained_remat_num_layers)

        grads = grad_func(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        # TODO(lmzheng): add dynamic scaling for mixed-precision training
        return new_state

    return train_step


def prepare_gpt_bert_input_and_model(model_type,
                                     benchmark_case,
                                     add_manual_remat=None,
                                     add_manual_layer_marker=None,
                                     num_manual_pipeline_stages=None,
                                     aval_train_state=True,
                                     tie_word_embeddings=False):
    print_used_time(None)
    batch_size = benchmark_case.batch_size
    (seq_len, hidden_size, num_layers, num_heads,
     vocab_size) = benchmark_case.model_config
    dtype = jnp.float16
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
    if model_type == "bert":
        model = FlaxBertForMaskedLMModule(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
            tie_word_embeddings=tie_word_embeddings,
            gradient_checkpointing=add_manual_remat,
            add_manual_pipeline_markers=add_manual_layer_marker,
            pipeline_mp_size=num_manual_pipeline_stages,
        ),
                                          dtype=dtype)
    elif model_type == "gpt":
        model = FlaxGPTForLMModule(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
            tie_word_embeddings=tie_word_embeddings,
            gradient_checkpointing=add_manual_remat,
            add_manual_pipeline_markers=add_manual_layer_marker,
            pipeline_mp_size=num_manual_pipeline_stages,
        ),
                                   dtype=dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    if aval_train_state:
        state = create_train_state_aval(rngkey, model, batch, dtype)
    else:
        state = create_train_state(rngkey, model, batch, dtype)
    print_used_time("Create train state")
    return state, batch, rngkey


def compute_gpt_bert_statistics(benchmark_case, latencies, num_devices):
    batch_size = benchmark_case.batch_size
    (seq_len, hidden_size, num_layers, num_heads,
     vocab_size) = benchmark_case.model_config

    tflops = compute_gpt_tflops(batch_size, seq_len, num_layers, hidden_size,
                                vocab_size, num_devices, np.mean(latencies))
    tflops_ckpt = compute_gpt_tflops(batch_size, seq_len, num_layers,
                                     hidden_size, vocab_size, num_devices,
                                     np.mean(latencies), True)
    parameter_count = compute_gpt_parameter_count(num_layers, hidden_size,
                                                  vocab_size)
    return tflops, tflops_ckpt, parameter_count


def benchmark_gpt_bert_3d_internal(model_type,
                                   benchmark_case,
                                   niter,
                                   num_hosts,
                                   num_devices_per_host,
                                   aval_train_state=True):
    # Connect to the cluster
    virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
        host_ids=list(range(num_hosts)),
        num_devices_per_host=num_devices_per_host)
    set_global_virtual_physical_mesh(virtual_mesh)

    # Parallel configs
    if benchmark_case.parallel_mode == "load_solution":
        use_fine_grained_remat = benchmark_case.parallel_args.use_remat
        fine_grained_remat_num_layers = benchmark_case.model_config.num_layers
    else:
        use_fine_grained_remat = None
        fine_grained_remat_num_layers = None
    (method, add_manual_remat, add_manual_layer_marker,
     num_manual_pipeline_stages) = get_pipeshard_parallel_method(
         benchmark_case,
         virtual_mesh.num_devices_per_host,
         use_fine_grained_remat=use_fine_grained_remat)

    state, batch, rngkey = prepare_gpt_bert_input_and_model(
        model_type,
        benchmark_case,
        add_manual_remat=add_manual_remat,
        add_manual_layer_marker=add_manual_layer_marker,
        num_manual_pipeline_stages=num_manual_pipeline_stages,
        aval_train_state=aval_train_state)

    train_step = get_train_step(method, use_fine_grained_remat,
                                fine_grained_remat_num_layers)

    (latencies, max_mem_allocated, compilation_times,
     executable) = compile_and_benchmark_pipeshard_executable(
         benchmark_case.parallel_mode, niter, train_step, state,
         (batch, rngkey))

    tflops, tflops_ckpt, parameter_count = compute_gpt_bert_statistics(
        benchmark_case, latencies, virtual_mesh.num_devices)

    # report_pipeline_breakdown(executable,
    #                           ["resharding_send", "resharding_recv",
    #                            "compute"],
    #                           niter)

    return (parameter_count, max_mem_allocated, latencies, tflops, tflops_ckpt,
            compilation_times) + get_last_dp_result()


def benchmark_gpt_bert_2d_internal(physical_mesh, model_type, benchmark_case,
                                   niter):
    # Model configs
    method, grad_func, use_remat = get_shard_parallel_method(
        benchmark_case, physical_mesh)

    state, batch, rngkey = prepare_gpt_bert_input_and_model(
        model_type,
        benchmark_case,
        add_manual_remat=use_remat,
        aval_train_state=global_config.use_dummy_value_for_benchmarking)

    # Compile executable
    train_step = get_train_step(method, grad_func=grad_func)

    (latencies, ilp_objective, alloc_mem,
     executable) = compile_and_benchmark_shard_executable(
         physical_mesh, niter, train_step, state, (batch, rngkey))

    # Compute statistics
    tflops, tflops_ckpt, parameter_count = compute_gpt_bert_statistics(
        benchmark_case, latencies, physical_mesh.num_devices)
    if use_remat:
        tflops = tflops_ckpt
    peak_mem = max(physical_mesh.get_max_memory_allocated(), alloc_mem)
    return parameter_count, ilp_objective, peak_mem, latencies, tflops

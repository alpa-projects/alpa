"""Benchmark one case of inter-op + intra-op parallelism."""
import jax
import jax.numpy as jnp
import numpy as np
import optax

import alpa
from alpa import (parallelize, get_global_cluster,
                  set_global_virtual_physical_mesh)
from alpa.model.bert_model import BertConfig, FlaxBertForMaskedLMModule
from alpa.model.model_util import TrainState
from alpa.model.gpt_model import FlaxGPTForLMModule
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.util import print_used_time

from util import compute_gpt_parameter_count, compute_gpt_tflops
from benchmark_parallel_utils import (
    get_pipeshard_parallel_method,
    compile_and_benchmark_pipeshard_training_executable)


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
    use_master_copy = (dtype == jnp.float16)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              use_master_copy=use_master_copy,
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
    use_master_copy = (dtype == jnp.float16)
    state = TrainState.create_aval(apply_fn=model.apply,
                                   params=params,
                                   tx=tx,
                                   use_master_copy=use_master_copy,
                                   dynamic_scale=None)
    return state


def get_train_step(parallel_method, grad_func=None):

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

    bert_config = BertConfig(
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
    )

    # Init train state
    if model_type == "bert":
        model = FlaxBertForMaskedLMModule(bert_config, dtype=dtype)
    elif model_type == "gpt":
        model = FlaxGPTForLMModule(bert_config, dtype=dtype)
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
    use_remat = benchmark_case.parallel_args.use_remat

    tflops = compute_gpt_tflops(batch_size,
                                seq_len,
                                num_layers,
                                hidden_size,
                                vocab_size,
                                num_devices,
                                np.mean(latencies),
                                checkpoint_activations=use_remat)
    parameter_count = compute_gpt_parameter_count(num_layers, hidden_size,
                                                  vocab_size)
    return tflops, parameter_count


def benchmark_gpt_bert_3d_internal(model_type,
                                   benchmark_case,
                                   pipeline_schedule,
                                   niter,
                                   num_hosts,
                                   num_devices_per_host,
                                   aval_train_state=True,
                                   profile_driver_time=False):
    # Connect to the cluster
    virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
        host_ids=list(range(num_hosts)),
        num_devices_per_host=num_devices_per_host)
    set_global_virtual_physical_mesh(virtual_mesh)

    # Parallel configs
    (method, add_manual_remat, add_manual_layer_marker,
     num_manual_pipeline_stages) = get_pipeshard_parallel_method(
         benchmark_case,
         virtual_mesh.num_devices_per_host,
         use_fine_grained_remat=True,
         pipeline_schedule=pipeline_schedule)

    state, batch, rngkey = prepare_gpt_bert_input_and_model(
        model_type,
        benchmark_case,
        add_manual_remat=add_manual_remat,
        add_manual_layer_marker=add_manual_layer_marker,
        num_manual_pipeline_stages=num_manual_pipeline_stages,
        aval_train_state=aval_train_state)

    train_step = get_train_step(method)

    (latencies, max_mem_allocated, compilation_times,
     executable) = compile_and_benchmark_pipeshard_training_executable(
         benchmark_case.parallel_mode,
         niter,
         train_step,
         state, (batch, rngkey),
         profile_driver_time=profile_driver_time)

    tflops, parameter_count = compute_gpt_bert_statistics(
        benchmark_case, latencies, virtual_mesh.num_devices)

    # report_pipeline_breakdown(executable,
    #                           ["resharding_send", "resharding_recv",
    #                            "compute"],
    #                           niter)

    (compute_cost_file_name, forward_stage_layer_ids, submesh_shapes,
     logical_mesh_shapes, autosharding_option_dicts) = get_last_dp_result()
    metadata = {
        "compilation_times": compilation_times,
        "compute_cost_file_name": compute_cost_file_name,
        "forward_stage_layer_ids": forward_stage_layer_ids,
        "submesh_shapes": submesh_shapes,
        "logical_mesh_shapes": logical_mesh_shapes,
        "autosharding_option_dicts": autosharding_option_dicts,
    }

    return parameter_count, max_mem_allocated, latencies, tflops, metadata

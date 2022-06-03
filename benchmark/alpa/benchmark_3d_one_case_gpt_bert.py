"""Benchmark one case of inter-op + intra-op parallelism."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray
import time

import alpa
from alpa import (parallelize, global_config, get_global_cluster,
                  set_global_virtual_physical_mesh,
                  PipeshardParallel, ManualPipeshardParallel,
                  AutoShardingOption,
                  manual_layer_construction,
                  automatic_layer_construction, automatic_remat)
from alpa.model.bert_model import BertConfig, FlaxBertForMaskedLMModule
from alpa.model.model_util import TrainState
from alpa.model.gpt_model import FlaxGPTForLMModule
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.timer import timers
from alpa.util import print_used_time, to_str_round, GB

from benchmark.util import compute_gpt_parameter_count, compute_gpt_tflops



def report_pipeline_breakdown(executable, timer_names, niter):
    overall_costs = executable.get_execution_time_costs(warmup=0, timer_name="overall")

    print(">>> overall: {}...".format(overall_costs))
    other_percentage = [100.0] * niter
    other = overall_costs
    for timer_name in timer_names:
        costs = executable.get_execution_time_costs(warmup=0, timer_name=timer_name)
        if len(costs) == 0:
            costs = [0.0] * niter
        percentage = [cost / overall_costs[i] * 100 for i, cost in enumerate(costs)]
        other = [remain - costs[i] for i, remain in enumerate(other)]
        other_percentage = [remain - percentage[i] for i, remain in enumerate(other_percentage)]
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


def create_train_state_aval(rngkey, model, batch, dtype):
    params = jax.eval_shape(model.init, rngkey, batch["input_ids"],
                            batch["attention_mask"], batch["token_type_ids"],
                            batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax.chain(
        #optax.clip_by_global_norm(1.0),  # TODO(lmzheng): fix reduce-scatter for this
        optax.adamw(learning_rate=1e-2, mask=weight_decay_mask)
    )
    mixed_precision = (dtype == jnp.float16)
    state = TrainState.create_aval(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        mixed_precision=mixed_precision,
        dynamic_scale=None)
    return state


def get_train_step(parallel_method,
                   auto_layer,
                   num_manual_pipeline_stages,
                   num_auto_layers,
                   auto_remat_mode,
                   num_auto_remat_layers):

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
            label_mask = jnp.where(batch["labels"]  > 0, 1.0, 0.0)
            labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
            loss = - jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            loss = (label_mask * loss).sum() / label_mask.sum()
            return loss

        if not auto_layer:
            loss_func = manual_layer_construction(loss_func)
        else:
            if auto_remat_mode == "fine_grained":
                loss_func = automatic_remat(loss_func, layer_num=num_auto_remat_layers)
                loss_func = automatic_layer_construction(loss_func, layer_num=num_auto_layers)
            else:
                use_remat = True if auto_remat_mode is "coarse_grained" else False
                loss_func = automatic_layer_construction(loss_func,
                                                         remat_layer=use_remat,
                                                         layer_num=num_auto_layers)

        grads = alpa.grad(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        # TODO(lmzheng): add dynamic scaling for mixed-precision training
        return new_state

    return train_step


def benchmark_gpt_bert_internal(model_type, benchmark_case, niter,
                                num_hosts, num_devices_per_host, aval_train_state=True):
    print_used_time(None)

    # Model configs
    (batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size, num_micro_batches,
     parallel_mode, parallel_args) = benchmark_case
    dtype = jnp.float16
    tie_word_embeddings = False

    # Connect to the cluster
    virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
        host_ids=list(range(num_hosts)),
        num_devices_per_host=num_devices_per_host)
    set_global_virtual_physical_mesh(virtual_mesh)

    # Parallel configs
    if parallel_mode == "search":
        prefer_reduce_scatter, use_remat, num_auto_layers, auto_stage_option = parallel_args
        auto_layer = True
        auto_remat_mode = "coarse_grained" if use_remat else None
        num_auto_remat_layers = None
        add_manual_layer_marker = add_manual_remat = num_manual_pipeline_stages = False
        method = PipeshardParallel(
            stage_mode="auto",
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
               prefer_reduce_scatter=prefer_reduce_scatter),
            **auto_stage_option)
    elif parallel_mode == "load_solution":
        prefer_reduce_scatter, use_remat, num_auto_layers, manual_stage_option = parallel_args
        auto_layer = True
        auto_remat_mode = "fine_grained" if use_remat else None
        num_auto_remat_layers = num_layers
        add_manual_layer_marker = add_manual_remat = num_manual_pipeline_stages = False
        method = ManualPipeshardParallel(
            *manual_stage_option,
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter))
    elif parallel_mode == "manual":
        (prefer_reduce_scatter, use_remat, (dp, op, pp),
            force_batch_dim_mapping) = parallel_args
        as_option = AutoShardingOption(prefer_reduce_scatter=prefer_reduce_scatter)
        if force_batch_dim_mapping:
            as_option.force_batch_dim_to_mesh_dim = 0
        auto_layer = False
        num_auto_layers = auto_remat_mode = num_auto_remat_layers = None
        add_manual_layer_marker = True
        add_manual_remat = use_remat

        logical_mesh_shape = (dp, op)
        num_manual_pipeline_stages = pp
        num_mesh_devices = np.prod(logical_mesh_shape)
        num_devices_per_host = virtual_mesh.num_devices_per_host
        if num_mesh_devices <= num_devices_per_host:
            physical_mesh_shape = (1, num_mesh_devices)
        else:
            assert num_mesh_devices % num_devices_per_host == 0
            physical_mesh_shape = (num_mesh_devices // num_devices_per_host,
                                   num_devices_per_host)

        method = ManualPipeshardParallel(
            num_micro_batches=num_micro_batches,
            forward_stage_layer_ids=[[i] for i in range(pp)],
            submesh_physical_shapes=[physical_mesh_shape] * pp,
            submesh_logical_shapes=[logical_mesh_shape] * pp,
            submesh_autosharding_option_dicts=[{}] * pp,
            default_auto_sharding_option=as_option)
    else:
        raise ValueError(f"Invalid model: {parallel_mode}")

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
        ), dtype=dtype)
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
        ), dtype=dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    if aval_train_state:
        state = create_train_state_aval(rngkey, model, batch, dtype)
    else:
        state = create_train_state(rngkey, model, batch, dtype)
    print_used_time("Create train state")

    # Compile executable
    train_step = get_train_step(method, auto_layer, num_manual_pipeline_stages,
                                num_auto_layers, auto_remat_mode,
                                num_auto_remat_layers)
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    if parallel_mode == "search":
        compilation_times = {k : timers(k).elapsed() for k in
                ["stage-construction", "stage-construction-dp",
                 "stage-construction-compilation", "stage-construction-profiling"]}
        print(f"compilation time breakdown: {to_str_round(compilation_times, 2)}")
    else:
        compilation_times = None

    # Dump hlo ir for debugging
    stage_hlo_texts = executable.get_hlo_text()
    for i in range(len(stage_hlo_texts)):
        with open(f"tmp/stage_{i}.hlo", "w") as fout:
            fout.write(stage_hlo_texts[i])
    with open(f"tmp/resharding_tasks.txt", "w") as fout:
        fout.write(executable.print_resharding_tasks())

    executable.sync()
    print_used_time("Compile (worker)")

    # Benchmark latency without driver overhead
    for i in range(niter):
        print(f"Iteration {i} ...")
        state = train_step(state, batch, rngkey)
        executable.sync()

    latencies = executable.get_execution_time_costs(warmup=1)
    max_mem_allocated = executable.mesh_group.get_max_memory_allocated()

    # Benchmark latency with driver overhead
    if False:
        global_config.use_dummy_value_for_benchmarking = False
        global_config.pipeline_sync_for_timer = False
        number = niter
        executable.sync()
        tic = time.time()
        for i in range(number):
            state = train_step(state, batch, rngkey)
        executable.sync()
        e2e_latency = (time.time() - tic) / number
        print(f"latency with dirver overhead: {e2e_latency:.3f}")
    print_used_time("Benchmark")

    # Compute statistics
    tflops = compute_gpt_tflops(batch_size, seq_len, num_layers,
                                hidden_size, vocab_size,
                                virtual_mesh.num_devices,
                                np.mean(latencies))
    tflops_ckpt = compute_gpt_tflops(batch_size, seq_len, num_layers,
                                     hidden_size, vocab_size,
                                     virtual_mesh.num_devices,
                                     np.mean(latencies), True)
    parameter_count = compute_gpt_parameter_count(num_layers, hidden_size, vocab_size)
    #report_pipeline_breakdown(executable, ["resharding_send", "resharding_recv", "compute"], niter)
    return (parameter_count, max_mem_allocated, latencies,
            tflops, tflops_ckpt, compilation_times) + get_last_dp_result()

"""Benchmark one case of inter-op + intra-op parallelism."""
from functools import partial

from flax import linen as nn, optim
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

import parax
from parax import (parallelize, global_config, set_parallelize_options, testing,
                   DeviceCluster, automatic_layer_construction)
from parax.model.wide_resnet import get_wide_resnet, TrainState
from parax.pipeline_parallel.stage_construction import get_last_dp_result
from parax.timer import timers
from parax.util import map_to_shape, print_used_time, compute_param_number, GB, to_str_round

# B = batch_size, I = image_size,
# L = num_layers, C = num_base_channels, W = width_factor, 
# D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FM = force_batch_dim_mapping,
# RS = prefer_reduce_scatter, Remat = use_rematerialization
# LS = logical_mesh_search_space

paper_auto_wresnet_suite = {  # key = number of gpus, value = a list of cases
1: [
    # B,   I,   L,   C,    W,  dtype,  NB, FD,    RS,    Remat, LS
    (2048, 224, 50,  160,  2,  "fp32", 32, False, False, True,  "single_node_model_parallel"),
],

2: [
    # B,   I,   L,   C,    W,  dtype,  NB, FD,    RS,    Remat, LS
    (1536, 224, 50,  224,  2, "fp32",  32, False, True,  True,  "single_node_model_parallel"),
],

4 : [
    # B,   I,   L,   C,    W,  dtype,  NB, FD,    RS,    Remat, LS
   (1536,  224, 50,  320,  2, "fp32",  32, False, True,  True,  "single_node_model_parallel"),
],

8: [
    # B,   I,   L,   C,    W,  dtype,  NB, FD,    RS,    Remat, LS
    (1536, 224, 50,  448,  2, "fp32",  32, False, True,  True,  "single_node_model_parallel"),
],

16: [
    # B,   I,   L,   C,    W,  dtype,  NB, FD,    RS,    Remat, LS
    (1536, 224, 50,  640,  2,  "fp32", 32, False, True,  True,  "single_node_model_parallel"),
],

32: [
    # B,   I,   L,   C,    W,  dtype,  NB, FD,    RS,    Remat, LS
    #(1536, 224, 50,  640,  2,  "fp32", 32, False, True,  True, "single_node_model_parallel"),
    (1520, 224, 50,  320,  16, "fp32", 38, False, False, True,  "single_node_model_parallel"),
],
}


as_option = global_config.default_autosharding_option


def compute_metrics(logits, labels):
    metrics = {
        "loss": cross_entropy_loss(logits, labels),
        "accuracy": jnp.mean(jnp.argmax(logits, -1) == labels),
    }
    return metrics


def cross_entropy_loss(logits, labels):
    num_classes = logits.shape[-1]
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


def create_learning_rate_fn():
    """Create learning rate schedule."""
    base_learning_rate = 0.1
    warmup_epochs = 5.0
    steps_per_epoch = 10000
    num_epochs = 100.0

    warmup_fn = optax.linear_schedule(init_value=0.,
                                      end_value=base_learning_rate,
                                      transition_steps=warmup_epochs *
                                      steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,
                                            decay_steps=cosine_epochs *
                                            steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn


def create_train_state(rngkey, model, input_images, learning_rate_fn):
    params = model.init_dummy(rngkey, input_images)
    params, batch_stats = params["params"], params["batch_stats"]

    # dynamic_scale = optim.DynamicScale()
    dynamic_scale = None

    tx = optax.sgd(
        learning_rate=learning_rate_fn,
        momentum=0.9,
        nesterov=True,
    )
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              batch_stats=batch_stats,
                              dynamic_scale=None)
    return state


def get_train_step(learning_rate_fn, use_grad_acc, use_remat):

    @parallelize
    def train_step(state, batch):

        @partial(automatic_layer_construction,
                 layer_num=16,
                 remat_layer=use_remat)
        def loss_fn(params):
            logits, new_model_state = state.apply_fn(
                {
                    "params": params,
                    "batch_stats": state.batch_stats
                },
                batch["images"],
                mutable=["batch_stats"])
            loss = cross_entropy_loss(logits, batch["labels"])
            # weight_penalty_params = jax.tree_leaves(params)
            # weight_decay = 0.0001
            # weight_l2 = sum(
            #     [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
            # weight_penalty = weight_decay * 0.5 * weight_l2
            metrics = {
                "loss": loss,
                "accuracy": jnp.mean(jnp.argmax(logits, -1) == batch["labels"]),
                "lr": learning_rate_fn(step)
            }
            return loss, (new_model_state, metrics)

        step = state.step
        dynamic_scale = state.dynamic_scale

        if dynamic_scale:
            # TOOD(lmzheng): handle gradient accumulation for this
            grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
            dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
            # dynamic loss takes care of averaging gradients across replicas
        else:
            if use_grad_acc:
                get_grad_fn = parax.grad
            else:
                get_grad_fn = jax.grad

            grad_fn = get_grad_fn(loss_fn, has_aux=True)
            grads, aux = grad_fn(state.params)
        new_model_state, metrics = aux

        new_state = state.apply_gradients(
            grads=grads, batch_stats=new_model_state["batch_stats"])
        if dynamic_scale:
            # if is_fin == False the gradients contain Inf/NaNs and optimizer
            # state and params should be restored (= skip this step).
            new_state = new_state.replace(
                opt_state=jax.tree_multimap(partial(jnp.where, is_fin),
                                            new_state.opt_state,
                                            state.opt_state),
                params=jax.tree_multimap(partial(jnp.where, is_fin),
                                         new_state.params, state.params))
            metrics["scale"] = dynamic_scale.scale

        return new_state, metrics

    return train_step


def benchmark_wresnet_internal(benchmark_case, niter,
                               num_hosts, num_devices_per_host):
    print_used_time(None)

    # Model configs
    model_type = "wide_resnet"
    batch_size, image_size, num_layers, num_channels, width_factor, dtype,\
        num_micro_batches, force_batch_dim_mapping,\
        prefer_reduce_scatter, use_remat, logical_mesh_search_space = benchmark_case
    if dtype == "fp32":
        dtype = jnp.float32
    elif dtype == "fp16":
        dtype = jnp.float16
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    # Parallel configs
    if num_micro_batches > 1:
        use_grad_acc = True
    else:
        use_grad_acc = False
        num_micro_batches = None

    as_option.prefer_reduce_scatter = prefer_reduce_scatter
    as_option.allow_mixed_mesh_shape = True

    device_cluster = DeviceCluster()
    host_ids = None if num_hosts == None else list(range(num_hosts))
    virtual_mesh = device_cluster.get_virtual_physical_mesh(
        host_ids=host_ids, num_devices_per_host=num_devices_per_host)
    set_parallelize_options(devices=virtual_mesh,
                            strategy="3d_parallel",
                            num_micro_batches=num_micro_batches,
                            pipeline_stage_mode="auto_gpipe",
                            #cache_compute_cost="compute-cost-2022-01-14-12-35-01.npy",
                            logical_mesh_search_space=logical_mesh_search_space)
    global_config.auto_stage_construction_imbalance_tolerance = 0.4

    # Prepare input batch
    num_classes = 1024
    batch = {
        "images": jnp.ones((batch_size, image_size, image_size, 3),
                           dtype=dtype),
        "labels": jnp.ones((batch_size), dtype=jnp.int32),
    }
    print_used_time("Prepare input")

    # Init train state
    if model_type == "wide_resnet":
        model = get_wide_resnet(num_layers, width_factor, num_channels,
                                num_classes, dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    learning_rate_fn = create_learning_rate_fn()
    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, batch["images"], learning_rate_fn)
    train_step = get_train_step(learning_rate_fn, use_grad_acc, use_remat)
    print_used_time("Create train state")
    parameter_count = compute_param_number(state.params)

    # Compile executable
    executable = train_step.get_executable(state, batch)
    print_used_time("Compile (driver)")

    compilation_times = {k : timers(k).elapsed() for k in
            ["stage-construction", "stage-construction-dp",
             "stage-construction-compilation", "stage-construction-profiling"]}
    print(f"compilation time breakdown: {to_str_round(compilation_times, 2)}")

    # Dump hlo ir for debugging
    stage_hlo_texts = executable.get_hlo_text()
    for i in range(len(stage_hlo_texts)):
        with open(f"tmp/stage_{i}.hlo", "w") as fout:
            fout.write(stage_hlo_texts[i])
    with open(f"tmp/resharding_tasks.txt", "w") as fout:
        fout.write(executable.print_resharding_tasks())

    executable.sync()
    print_used_time("Compile (workers)")

    # Benchmark step time
    for i in range(niter):
        state, metrics = train_step(state, batch)

    # for timer_name in ["resharding_send", "resharding_recv", "compute"]:
    #     latencies = executable.get_execution_time_costs(warmup=2, timer_name=timer_name, return_all_costs=True)
    #     print(f"{timer_name}: ")
    #     for i, t in enumerate(latencies):
    #         pstr = f"Mesh {i}: "
    #         pstr += f"{np.mean(t)}s. Each iter: {t}"
    #         print(pstr)
    latencies = executable.get_execution_time_costs(warmup=2)
    print_used_time("Benchmark")

    mem_allocated = executable.get_memory_allocated()
    max_mem_allocated = executable.get_max_memory_allocated()

    # Compute statistics
    num_gpus = virtual_mesh.num_devices
    tflops = executable.flop_count / num_gpus / np.mean(latencies) / 1e12
    del state
    del metrics
    for i, profiled in enumerate(executable.profile_all_executables()):
        pstr = f"Mesh {i}: "
        for k in profiled:
            pstr += f"Exec {k}: {profiled[k][0]}s; "
        print(pstr)
    executable.shutdown()

    return (parameter_count, mem_allocated, max_mem_allocated, latencies,
            tflops, tflops, compilation_times) + get_last_dp_result()

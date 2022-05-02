"""Benchmark one case of intra-op only parallelism."""
from flax import linen as nn, optim
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax

import alpa
from alpa import parallelize, global_config, set_parallelize_options
from alpa.model.wide_resnet import get_wide_resnet, TrainState
from alpa.util import (map_to_shape, count_communication_primitives,
                        print_used_time, compute_param_number, GB)

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

    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=base_learning_rate,
        transition_steps=warmup_epochs * steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch)
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
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        dynamic_scale=None)
    return state


def get_train_step(learning_rate_fn, use_grad_acc):

    @parallelize
    def train_step(state, batch):
        def loss_fn(params):
            logits, new_model_state = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                batch["images"],
                mutable=["batch_stats"])
            loss = cross_entropy_loss(logits, batch["labels"])
            weight_penalty_params = jax.tree_leaves(params)
            weight_decay = 0.0001
            weight_l2 = sum([jnp.sum(x ** 2)
                             for x in weight_penalty_params
                             if x.ndim > 1])
            weight_penalty = weight_decay * 0.5 * weight_l2
            metrics = {
              "loss": loss,
              "accuracy": jnp.mean(jnp.argmax(logits, -1) == batch["labels"]),
              "lr": learning_rate_fn(step)
            }
            return loss + weight_penalty, (new_model_state, metrics)

        step = state.step
        dynamic_scale = state.dynamic_scale

        if dynamic_scale:
            # TOOD(lmzheng): handle gradient accumulation for this
            grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
            dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
            # dynamic loss takes care of averaging gradients across replicas
        else:
            if use_grad_acc:
                get_grad_fn = alpa.grad
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
                opt_state=jax.tree_multimap(
                    partial(jnp.where, is_fin),
                    new_state.opt_state,
                    state.opt_state),
                params=jax.tree_multimap(
                    partial(jnp.where, is_fin),
                    new_state.params,
                    state.params))
            metrics["scale"] = dynamic_scale.scale

        return new_state, metrics

    return train_step


def benchmark_wresnet_internal(physical_mesh, benchmark_case, niter):
    print_used_time(None)

    # Model configs
    model_type = "wresnet"
    (batch_size, image_size, num_layers, num_channels, width_factor, dtype,
     num_micro_batches, parallel_mode, parallel_args) = benchmark_case
    if dtype == "fp32":
        dtype = jnp.float32
    elif dtype == "fp16":
        dtype = jnp.float16
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    # Parallel configs
    (prefer_reduce_scatter, use_remat, logical_mesh_shape,
     force_batch_dim_mapping) = parallel_args

    if num_micro_batches > 1:
        use_grad_acc = True
    else:
        use_grad_acc = False
        num_micro_batches = None

    if force_batch_dim_mapping:
        # Always map batch dim to mesh dim 0
        as_option.force_batch_dim_to_mesh_dim = 0
    as_option.prefer_reduce_scatter = prefer_reduce_scatter
    as_option.allow_mixed_mesh_shape = True
    if parallel_mode == "zero-3":
        as_option.force_zero_stage_3 = True
    elif parallel_mode in ["shard-largest"]:
        as_option.force_simple_heuristic = other
        global_config.remat_using_while = True

    logical_mesh = physical_mesh.get_logical_mesh(logical_mesh_shape,
                                                  mesh_topology="tree",
                                                  inter_host_bandwidth=1,
                                                  intra_host_bandwidth=30)
    set_parallelize_options(devices=logical_mesh,
                            num_micro_batches=num_micro_batches)
    print_used_time("Setup device mesh")

    # Prepare input batch
    num_classes = 1024
    batch = {
        "images": jnp.ones((batch_size, image_size, image_size, 3), dtype=dtype),
        "labels": jnp.ones((batch_size), dtype=jnp.int32),
    }
    print_used_time("Prepare input")

    # Init train state
    if model_type == "wresnet":
        model = get_wide_resnet(num_layers, width_factor,
                                num_channels, num_classes,
                                dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    learning_rate_fn = create_learning_rate_fn()
    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, batch["images"], learning_rate_fn)
    param_count = compute_param_number(state.params)
    train_step = get_train_step(learning_rate_fn, use_grad_acc)
    print_used_time("Create train state")

    # Compile executable
    executable = train_step.get_executable(state, batch)
    print_used_time("Compile (driver)")

    physical_mesh.sync_workers()
    print_used_time("Compile (workers)")

    # Check sharding strategy
    alloc_mem = executable.get_total_allocation_size()
    ilp_objective = executable.auto_sharding_objective or 0.0
    hlo_text = executable.get_hlo_text()
    with open("tmp/last_wresnet_2d.hlo", "w") as fout:
        fout.write(hlo_text)
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all =\
        count_communication_primitives(hlo_text)

    print(f"#total: {n_total}, #all-reduce: {n_all_reduce}, "
          f"#all-gather: {n_all_gather}, #reduce-scatter: {n_reduce_scatter}, "
          f"#all-to-all: {n_all_to_all}")
    print(f"alloc_mem: {alloc_mem / GB:.2f} GB")

    # Benchmark step time
    warmup = 2 if niter >= 5 else 1

    if alloc_mem > physical_mesh.get_available_memory():
        latencies = [-1]
    else:
        for i in range(niter):
            print(f"Iteration {i}")
            state, metrics = train_step(state, batch)

        latencies = executable.get_execution_time_costs(warmup=warmup)
    print_used_time("Benchmark")

    # Compute statistics
    num_gpus = physical_mesh.num_devices
    tflops = executable.flop_count / num_gpus / np.mean(latencies) / 1e12
    peak_mem = max(physical_mesh.get_max_memory_allocated(), alloc_mem)

    return param_count, ilp_objective, peak_mem, latencies, tflops

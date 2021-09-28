import argparse
from functools import partial
import time

from flax import linen as nn, optim
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import ray
import optax

import parax
from parax import (parallelize, global_config, set_parallelize_options, testing,
                   DeviceCluster, PhysicalDeviceMesh, automatic_layer_slicing)
from parax.model.wide_resnet import get_wide_resnet, TrainState
from parax.util import (run_cmd, write_tsv, map_to_shape, list_gpu_info,
                        count_communication_primitives, print_used_time,
                        compute_param_number)


GB = 1024 ** 3


def compute_metrics(logits, labels):
    metrics = {
        "loss": cross_entropy_loss(logits, labels),
        "accuracy": jnp.mean(jnp.argmax(logits, -1) == labels),
    }
    return metrics


def cross_entropy_loss(logits, labels):
    one_hot_labels = common_utils.onehot(labels, num_classes=1000)
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


def train_step_func(state, batch, learning_rate_fn, use_grad_acc):
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
            get_grad_fn = parax.grad
        else:
            get_grad_fn = jax.grad

        grad_fn = get_grad_fn(loss_fn, has_aux=True)
        grads, aux = grad_fn(state.params)
    new_model_state, metrics = aux

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"])
    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        new_state = new_state.replace(
            opt_state=jax.tree_multimap(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state,
                state.opt_state),
            params=jax.tree_multimap(
                functools.partial(jnp.where, is_fin),
                new_state.params,
                state.params))
        metrics["scale"] = dynamic_scale.scale

    return new_state, metrics


def benchmark_model_one_case(benchmark_case):
    print_used_time(None)

    # Model configs
    model_type = args.model
    batch_size, image_size, num_layers, num_channels, width_factor, dtype,\
        mesh_dim0, mesh_dim1, num_micro_batches, force_data_parallel,\
        use_remat = benchmark_case
    if dtype == "fp32":
        dtype = jnp.float32
    elif dtype == "fp16":
        dtype = jnp.float16
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    # Parallel configs
    global_config.force_data_parallel = force_data_parallel

    if num_micro_batches > 1:
        use_grad_acc = True
        global_config.prefer_reduce_scatter = False
    else:
        use_grad_acc = False
        global_config.prefer_reduce_scatter = False
        num_micro_batches = None

    if args.local:
        physical_mesh = PhysicalDeviceMesh(jax.devices())
    else:
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
    logical_mesh = physical_mesh.get_logical_mesh([mesh_dim0, mesh_dim1],
                                                  mesh_topology="tree",
                                                  inter_host_bandwidth=1,
                                                  intra_host_bandwidth=30)
    set_parallelize_options(devices=logical_mesh,
                            num_micro_batches=num_micro_batches)
    print_used_time("Setup device mesh")

    # Prepare input batch
    num_classes = 1000
    batch = {
        "images": jnp.ones((batch_size, image_size, image_size, 3), dtype=dtype),
        "labels": jnp.ones((batch_size), dtype=jnp.int32),
    }
    print_used_time("Prepare input")

    # Init train state
    if model_type == "wide_resnet":
        model = get_wide_resnet(num_layers, width_factor,
                                num_channels, num_classes,
                                dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    learning_rate_fn = create_learning_rate_fn()
    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, batch["images"], learning_rate_fn)
    train_step = partial(train_step_func, learning_rate_fn=learning_rate_fn,
                         use_grad_acc=use_grad_acc)
    train_step = parallelize(train_step)
    print_used_time("Create train state")
    param_count = compute_param_number(state.params)

    # Compile executable
    executable = train_step.get_executable(state, batch)
    print_used_time("Compile (driver)")

    physical_mesh.sync_workers()
    print_used_time("Compile (workers)")

    # Benchmark step time
    for i in range(args.niter):
        state, metrics = train_step(state, batch)

    costs = executable.get_execution_time_costs(warmup=2)
    print_used_time("Benchmark")

    # Check sharding strategy
    objective = testing.last_compiled_auto_sharding_objective or 0.0
    alloc_mem = executable.get_total_allocation_size()
    hlo_text = executable.get_hlo_text()

    with open("last.hlo", "w") as fout:
        fout.write(hlo_text)
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all =\
        count_communication_primitives(hlo_text)
    print(f"#total: {n_total}, #all-reduce: {n_all_reduce}, "
          f"#all-gather: {n_all_gather}, #reduce-scatter: {n_reduce_scatter}, "
          f"#all-to-all: {n_all_to_all}")

    # Log benchmark results
    num_gpus = mesh_dim0 * mesh_dim1
    tflops = executable.flop_count / num_gpus / np.mean(costs) / 1e12
    heads = ["Model", "Model Config", "Parallel Config", "Param count", 
             "Alloc Mem", "ILP Objective", "Mean Time", "Std Time", "TFLOPS"]
    values = [model_type, str(benchmark_case[:-5]), str(benchmark_case[-5:]),
              f"{param_count/1e9:.3f}",
              f"{alloc_mem/GB:.3f}", f"{objective:.2f}",
              f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}",
              f"{tflops:.2f}"]
    write_tsv(heads, values, f"result_{model_type}.tsv")

    physical_mesh.shutdown()


# B = batch_size, I = image_size,
# L = num_layers, C = num_base_channels, W = width_factor, 
# D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FD = force_data_parallel, CK = use_checkpoint

default_benchmark_suite = {  # key = number of gpus, value = a list of cases
1: [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FD,    CK,
    (16,   224, 50,  256, 4, "fp32", 1,  1,  1,  False, False),
],

8: [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FD,    CK,
    (32,   224, 50,  512, 4, "fp32", 2,  4,  1,  False, False),
    (64,   224, 50,  512, 4, "fp32", 2,  4,  2,  False, False),
    (16,   224, 50,  704, 4, "fp32", 8,  1,  1,  False, False),
],

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
    parser.add_argument("--model", type=str, default="wide_resnet")
    parser.add_argument("--niter", type=int, default=4,
        help="Number of benchmark iteration")
    parser.add_argument("--suite", choices=["default"], default="default")
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    args = parser.parse_args()

    if not args.local:
        ray.init(address="auto")
        jax.config.update("jax_platform_name", "cpu")

    global_config.use_dummy_value_for_benchmarking = True

    benchmark_all()

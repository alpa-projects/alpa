import argparse
from datetime import datetime
from functools import partial
import pickle
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
from parax.pipeline_parallel.decentralized_distributed_runtime import DecentralizedDistributedRuntime
from parax.util import (run_cmd, write_tsv, map_to_shape, list_gpu_info,
                        count_communication_primitives, print_used_time,
                        compute_param_number)

GB = 1024**3

TMP_PICKLE_FILE_NAME = "tmp/tmp_transfer.pkl"


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

        @partial(automatic_layer_slicing,
                 layer_num=16,
                 use_pipeline=True,
                 use_remat=use_remat)
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


def benchmark_wide_resnet_internal(benchmark_case, niter,
                                   logical_mesh_search_space, num_hosts,
                                   num_devices_per_host):
    # Backup global config
    print_used_time(None)
    backup = global_config.backup()

    # Model configs
    model_type = "wide_resnet"
    batch_size, image_size, num_layers, num_channels, width_factor, dtype,\
        num_micro_batches, force_data_parallel,\
        prefer_reduce_scatter, use_remat = benchmark_case
    if dtype == "fp32":
        dtype = jnp.float32
    elif dtype == "fp16":
        dtype = jnp.float16
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    device_cluster = DeviceCluster()
    host_ids = None if num_hosts == None else list(range(num_hosts))
    virtual_mesh = device_cluster.get_virtual_physical_mesh(
        host_ids=host_ids, num_devices_per_host=num_devices_per_host)
    set_parallelize_options(devices=virtual_mesh,
                            strategy="3d_parallel",
                            num_micro_batches=num_micro_batches,
                            pipeline_stage_mode="auto_gpipe",
                            logical_mesh_search_space=logical_mesh_search_space)

    # Parallel configs
    if num_micro_batches > 1:
        use_grad_acc = True
        global_config.prefer_reduce_scatter = False
    else:
        use_grad_acc = False
        num_micro_batches = None

    global_config.force_data_parallel = force_data_parallel
    global_config.prefer_reduce_scatter = prefer_reduce_scatter
    global_config.allow_mixed_mesh_shape = False
    global_config.auto_stage_construction_imbalance_tolerance = 0.4
    global_config.use_dummy_value_for_benchmarking = True
    global_config.use_scatter_gather = True

    print_used_time("Setup device mesh")

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
    param_count = compute_param_number(state.params)

    # Compile executable
    executable: DecentralizedDistributedRuntime = train_step.get_executable(
        state, batch)
    print_used_time("Compile (driver)")

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

    # Compute statistics
    num_gpus = virtual_mesh.total_devices
    tflops = executable.flop_count / num_gpus / np.mean(latencies) / 1e12
    peak_mem = executable.get_max_memory_allocated() / GB
    del state
    del metrics
    for i, profiled in enumerate(executable.profile_all_executables()):
        pstr = f"Mesh {i}: "
        for k in profiled:
            pstr += f"Exec {k}: {profiled[k][0]}s; "
        print(pstr)
    executable.shutdown()

    # Restore global config
    global_config.restore(backup)

    return latencies, peak_mem, tflops, param_count


def benchmark_wresnet_3d_one_case(case,
                                  niter,
                                  num_hosts,
                                  num_devices_per_host,
                                  use_separate_process=True,
                                  dump_result=False,
                                  logical_mesh_search_space="default"):
    if not use_separate_process:
        ray.init(address="auto", ignore_reinit_error=True)
        jax.config.update('jax_platform_name', 'cpu')
        global_config.use_dummy_value_for_benchmarking = True

        result = benchmark_wide_resnet_internal(case, niter,
                                                logical_mesh_search_space,
                                                num_hosts, num_devices_per_host)
    else:
        cmd = (f"python3 -u benchmark_wide_resnet_3d_one_case.py "
               f"--niter {niter} "
               f'--case "{case}" '
               f"--dump-result ")
        if num_hosts is not None:
            cmd += f"--num-hosts {num_hosts} "
        if num_devices_per_host is not None:
            cmd += f"--num-devices-per-host {num_devices_per_host} "
        cmd += f"--logical_mesh_search_space {logical_mesh_search_space}"
        # Run benchmark
        ret = run_cmd(cmd)
        if ret == 0:
            result = pickle.load(open(TMP_PICKLE_FILE_NAME, "rb"))
        else:
            result = -1, -1, -1, -1
    if dump_result:
        pickle.dump(result, open(TMP_PICKLE_FILE_NAME, "wb"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", type=int, default=4)
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--dump-result",
                        action="store_true",
                        help="Dump results into a temporary pickle file")
    parser.add_argument("--num-hosts", type=int, default=None)
    parser.add_argument("--num-devices-per-host", type=int, default=None)
    parser.add_argument(
        "--logical_mesh_search_space",
        choices=["single_node_model_parallel", "only_dp", "all", "default"],
        default="single_node_model_parallel",
        help="logical mesh search space in auto stage construction")
    args = parser.parse_args()

    run_cmd("mkdir -p tmp")
    case = eval(args.case)
    benchmark_wresnet_3d_one_case(
        case,
        args.niter,
        use_separate_process=False,
        dump_result=args.dump_result,
        num_hosts=args.num_hosts,
        num_devices_per_host=args.num_devices_per_host,
        logical_mesh_search_space=args.logical_mesh_search_space)

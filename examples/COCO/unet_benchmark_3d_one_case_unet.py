"""Benchmark one case of inter-op + intra-op parallelism."""
from functools import partial

from flax import linen as nn, optim
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

import alpa
from alpa import (parallelize, global_config, set_parallelize_options, testing,
                   DeviceCluster, automatic_layer_construction)
from alpa.model.wide_resnet import get_wide_resnet
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.timer import timers
from alpa.util import map_to_shape, print_used_time, compute_param_number, GB, to_str_round

from unet import UNet, TrainState
import utils

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



def create_train_state(rngkey, model, input, learning_rate_fn):
    """Create train state for unet."""
    params = model.init_dummy(rngkey, input, train=False)

    num_trainable_params = utils.compute_param_number(params)
    model_size = utils.compute_bytes(params)
    print(f"model size : {model_size/utils.MB} MB, num of trainable params : {num_trainable_params/1024/1024} M")

    tx = optax.sgd(learning_rate=learning_rate_fn)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              dynamic_scale=None)
                            #   batch_stats=batch_stats)
    return state


def block_cnt_to_pipeline_layer(channel_size, block_cnt, mp = dict()):
    if block_cnt[0] in mp:
        return mp[block_cnt[0]]
    return block_cnt[0]*2

def get_train_step_unet(learning_rate_fn, use_grad_acc, use_remat, num_layers=4):

    @parallelize(batch_argnums=(1,))
    def train_step(state, batch):
        """Train for a single step."""

        @partial(automatic_layer_construction,
                 layer_num=num_layers,
                 remat_layer=use_remat)
        def training_loss_fn(params):
            unet_out = state.apply_fn(
                params,
                batch["images"],
                train=True)
            label_one_hot = jax.nn.one_hot(batch["labels"], 12)
            loss = jnp.mean(optax.softmax_cross_entropy(logits=unet_out, labels=label_one_hot))
            
            metrics = {
                "loss": loss,
                "accuracy": jnp.mean(jnp.argmax(unet_out, -1) == batch["labels"]),
            }
            return loss, metrics

        if use_grad_acc:
            get_grad_fn = alpa.grad
        else:
            get_grad_fn = jax.grad

        grad_fn = get_grad_fn(training_loss_fn, has_aux=True)
        grads, metrics = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)#, batch_stats=new_model_state["batch_stats"])

        return new_state, metrics

    return train_step

def benchmark_unet_internal(benchmark_case, niter,
                               num_hosts, num_devices_per_host):
    as_option = global_config.default_autosharding_option
    print_used_time(None)

    # Model configs
    model_type = "unet"
    (batch_size, image_size, num_micro_batches, channel_size, block_cnt, 
    force_batch_dim_mapping, prefer_reduce_scatter, use_remat, logical_mesh_search_space) = benchmark_case

    print(f"(batch_size={batch_size} \nimage_size={image_size}\nnum_micro_batches={num_micro_batches}\nchannel_size={channel_size}\nblock_cnt={block_cnt}\nforce_batch_dim_mapping={force_batch_dim_mapping}\nprefer_reduce_scatter={prefer_reduce_scatter}\nuse_remat={use_remat}\nlogical_mesh_search_space={logical_mesh_search_space})")
    pipeline_stage_mode = "auto_gpipe"

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
                            strategy="pipeshard_parallel",
                            num_micro_batches=num_micro_batches,
                            pipeline_stage_mode=pipeline_stage_mode,
                            logical_mesh_search_space=logical_mesh_search_space,
                            # use_hlo_cost_model=True
                            )

    global_config.auto_stage_construction_imbalance_tolerance = 0.25

    ############ the above is config for alpa, the below is for running ############

    # Prepare input batch
    num_classes = 12
    rng = jax.random.PRNGKey(0)
    batch = {
        "images": jnp.ones((batch_size, image_size[0], image_size[1], 3), jnp.float32),
        "labels": jnp.ones((batch_size, image_size[0], image_size[1]), dtype=jnp.int32),
    }
    print_used_time("Prepare input")

    config = {
        "padding": "SAME",
        "use_batch_norm": False,
        "channel_size": channel_size,
        "block_cnt": block_cnt,
        "batch_size": batch_size,
        "learning_rate": 0.3,
        "tf_seed": 0,
        "train_epochs": 1
    }

    # Init train state
    if model_type == "unet":
        model = UNet(num_classes=num_classes,
                    padding=config['padding'],
                    use_batch_norm=config['use_batch_norm'],
                    channel_size=config['channel_size'],
                    block_cnt=config['block_cnt'])
    else:
        raise ValueError(f"Invalid model {model_type}")

    learning_rate_fn = create_learning_rate_fn()
    state = create_train_state(rng, model, batch["images"], learning_rate_fn)
    train_step = get_train_step_unet(learning_rate_fn, use_grad_acc, use_remat, num_layers=block_cnt_to_pipeline_layer(channel_size, block_cnt))
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
    # for i, profiled in enumerate(executable.profile_all_executables()):
    #     pstr = f"Mesh {i}: "
    #     for k in profiled:
    #         pstr += f"Exec {k}: {profiled[k][0]}s; "
    #     print(pstr)
    executable.shutdown()

    print(parameter_count, mem_allocated, max_mem_allocated, latencies,
            tflops, tflops, compilation_times)

    return (parameter_count, mem_allocated, max_mem_allocated, latencies,
            tflops, tflops, compilation_times) + get_last_dp_result()

"""The entry point of intra-op + inter-op parallelism benchmark."""
import argparse
from datetime import datetime
import time
import numpy as np
from alpa.util import write_tsv, get_num_hosts_and_num_devices, to_str_round, GB
import jax
import ray
from configs.benchmark_unet_suite import unet_suite
from alpa import global_config
from alpa.util import run_cmd, get_ray_namespace_str, disable_tqdm_globally
import tensorflow as tf
from functools import partial
from flax import linen as nn, optim
from flax.training import common_utils
import jax.numpy as jnp
import optax
import alpa
from alpa import (parallelize, global_config, set_parallelize_options, testing,
                   DeviceCluster, automatic_layer_construction)
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.timer import timers
from alpa.util import print_used_time, compute_param_number, to_str_round
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
    print(f"model size : {model_size/utils.MB} MB,"
          f"num of trainable params : {num_trainable_params/1024/1024} M")

    tx = optax.sgd(learning_rate=learning_rate_fn)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              dynamic_scale=None)
    return state


def block_cnt_to_pipeline_layer(channel_size, block_cnt, mp = dict()):
    if block_cnt[0] in mp:
        return mp[block_cnt[0]]
    return int(block_cnt[0]*1.5)

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
        new_state = state.apply_gradients(grads=grads)

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

    print(f"(batch_size={batch_size} \nimage_size={image_size}\nnum_micro_batches={num_micro_batches}\n"
          f"channel_size={channel_size}\nblock_cnt={block_cnt}\nforce_batch_dim_mapping={force_batch_dim_mapping}"
          f"nprefer_reduce_scatter={prefer_reduce_scatter}\nuse_remat={use_remat}\nlogical_mesh_search_space="
          f"{logical_mesh_search_space})")
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
    train_step = get_train_step_unet(learning_rate_fn, use_grad_acc, use_remat, 
                                     num_layers=block_cnt_to_pipeline_layer(channel_size, block_cnt))
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

    latencies = executable.get_execution_time_costs(warmup=2)
    print_used_time("Benchmark")

    mem_allocated = executable.get_memory_allocated()
    max_mem_allocated = executable.get_max_memory_allocated()

    # Compute statistics
    num_gpus = virtual_mesh.num_devices
    tflops = executable.flop_count / num_gpus / np.mean(latencies) / 1e12
    del state
    del metrics
    executable.shutdown()

    print(parameter_count, mem_allocated, max_mem_allocated, latencies,
            tflops, tflops, compilation_times)

    return (parameter_count, mem_allocated, max_mem_allocated, latencies,
            tflops, tflops, compilation_times) + get_last_dp_result()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", type=int, default=3,
        help="The number of benchmark iterations")
    parser.add_argument("--num-hosts", type=int, default=None)
    parser.add_argument("--num-devices-per-host", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default="default")
    args = parser.parse_args()

    disable_tqdm_globally()
    # Get the benchmark suite
    num_hosts, num_devices_per_host = get_num_hosts_and_num_devices(args)
    num_gpus = num_hosts * num_devices_per_host
    suite = unet_suite[num_gpus]

    model_type = "unet"
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_name = f"{model_type}_alpa_{args.exp_name}_{date_str}.tsv"

    # Run all cases
    for benchmark_case in suite:
        (batch_size, image_size, num_micro_batches, channel_size, block_cnt, 
        force_batch_dim_mapping, prefer_reduce_scatter, use_remat, logical_mesh_search_space) = benchmark_case
        model_config = (batch_size, image_size, channel_size, block_cnt)

        overwrite_global_config_dict = {}
        pipeline_stage_mode = "auto_gpipe"
        pipeline_mp_size = 1

        if pipeline_mp_size <= 1 and pipeline_stage_mode == "uniform_layer_gpipe":
            print(f"Skip the case: {str(benchmark_case)}, because PP <= 1. "
                  f"Please use `benchmark_2d.py` "
                  f"since 3d runtime will have a small overhead.")
            continue

        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))

        try:
            ray.init(address="auto", ignore_reinit_error=True,
                        namespace=get_ray_namespace_str())
            tf.config.experimental.set_visible_devices([], 'GPU')
            jax.config.update('jax_platform_name', 'cpu')
            global_config.use_dummy_value_for_benchmarking = True
            result = benchmark_unet_internal(benchmark_case, args.niter, num_hosts, num_devices_per_host)
            ray.shutdown()
        except:
            print("Fail ", model_config)
            result = -1, -1, -1, [-1], -1, -1, None, None, None, None, None, None
        
        (parameter_count, mem_allocated, max_mem_allocated, latencies, tflops,
         tflops_ckpt, compilation_times, compute_cost_file_name, forward_stage_layer_ids,
         submesh_shapes, logical_mesh_shapes, autosharding_option_dicts) = result

        heads = ["Type", "Model Config", "#GPUs", "#Layers (for Auto-Layer)",
                    "#Microbatch", "Remat", "Reduce-scatter",
                    "Mean Time", "Std Time", "#Params", "TFLOPs",
                    "TFLOPs (ckpt)", "Peak Mem", "Compute Cost File",
                    "Layer->Stage Mapping", "Submesh Shapes",
                    "Logical Mesh Shapes", "Autosharding Global Configs",
                    "overwrite_global_config_dict", "compilation times"]
        values = [model_type + "-" + pipeline_stage_mode, model_config, num_gpus, pipeline_mp_size,
                    num_micro_batches, use_remat, prefer_reduce_scatter,
                    f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                    f"{parameter_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops_ckpt:.2f}",
                    f"{max_mem_allocated/GB:.3f}G", compute_cost_file_name,
                    forward_stage_layer_ids, submesh_shapes,
                    logical_mesh_shapes, autosharding_option_dicts,
                    overwrite_global_config_dict, to_str_round(compilation_times, 2)]
        write_tsv(heads, values, output_name)

        time.sleep(0.1)  # for ctrl+c to work

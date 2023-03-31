"""Benchmark one case of inter-op + intra-op parallelism."""
from alpa.pipeline_parallel.layer_construction import ManualLayerOption
import jax
import jax.numpy as jnp
import numpy as np
import optax

import alpa
from alpa import (parallelize, get_global_cluster,
                  set_global_virtual_physical_mesh)
from alpa.model.unet_2d import get_unet_2d
from alpa.model.model_util import TrainState
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.util import print_used_time, compute_param_number
from benchmark_parallel_utils import (
    get_pipeshard_parallel_method,
    compile_and_benchmark_pipeshard_training_executable)


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


def create_train_state(rngkey, model, batch, learning_rate_fn):
    params = model.init_dummy(rngkey, *batch)

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
                              dynamic_scale=None)
    return state


def get_train_step(learning_rate_fn,
                   use_remat,
                   num_remat_layers,
                   method,
                   grad_func=None):

    if grad_func is None:
        grad_func = alpa.grad

    @parallelize(method=method)
    def train_step(state, batch):

        def loss_fn(params):
            outs = state.apply_fn(params, batch["images"], batch["timesteps"],
                                  batch["encoder_hidden_states"])
            sample = outs.sample
            loss = jnp.mean(
                optax.l2_loss(predictions=sample, targets=batch["targets"]))

            metrics = {"loss": loss, "lr": learning_rate_fn(step)}
            return loss, metrics

        step = state.step

        grad_fn = grad_func(loss_fn, has_aux=True)
        grads, aux = grad_fn(state.params)
        metrics = aux

        new_state = state.apply_gradients(grads=grads)

        return new_state, metrics

    return train_step


def prepare_unet_input_and_model(benchmark_case):
    print_used_time(None)
    # Model configs
    (batch_size, model_config, _, _, _) = benchmark_case
    (image_size, channel_size, block_cnt, dtype, _) = model_config
    in_channels = 3
    out_channels = 4

    # Prepare input batch
    encoder_factor = 2**(block_cnt - 1)
    # Unlike wide-resnet, we have a transpose of input image in unet 2d model.
    batch = {
        "images":
            jnp.ones((batch_size, in_channels, image_size, image_size),
                     dtype=dtype),
        "targets":
            jnp.ones((batch_size, out_channels, image_size, image_size),
                     dtype=dtype),
        "timesteps":
            1,
        "encoder_hidden_states":
            jnp.ones((batch_size, (image_size // encoder_factor)**2,
                      channel_size * encoder_factor // 2))
    }
    print_used_time("Prepare input")

    # Init train state

    down_block_types = (("CrossAttnDownBlock2D",) * (block_cnt - 1) +
                        ("DownBlock2D",))
    up_block_types = ("UpBlock2D",) + ("CrossAttnUpBlock2D",) * (block_cnt - 1)
    # Each downsampling, the num channels grows twice
    block_out_channels = [channel_size * (2**i) for i in range(block_cnt - 1)]
    block_out_channels.append(block_out_channels[-1])
    model = get_unet_2d(image_size,
                        down_block_types=down_block_types,
                        up_block_types=up_block_types,
                        block_out_channels=block_out_channels,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        layers_per_block=1,
                        dtype=dtype)

    rngkey = jax.random.PRNGKey(0)
    learning_rate_fn = create_learning_rate_fn()
    input_batch = (batch["images"], batch["timesteps"],
                   batch["encoder_hidden_states"])
    state = create_train_state(rngkey, model, input_batch, learning_rate_fn)
    print_used_time("Create train state")
    return state, batch, learning_rate_fn


def benchmark_unet_3d_internal(benchmark_case,
                               pipeline_schedule,
                               niter,
                               num_hosts,
                               num_devices_per_host,
                               profile_driver_time=False):
    # Connect to the cluster
    virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
        host_ids=list(range(num_hosts)),
        num_devices_per_host=num_devices_per_host)
    set_global_virtual_physical_mesh(virtual_mesh)

    # Parallel configs
    allow_mixed_mesh_shape = True
    (method, _, _, _) = get_pipeshard_parallel_method(
        benchmark_case,
        virtual_mesh.num_devices_per_host,
        allow_mixed_mesh_shape=allow_mixed_mesh_shape,
        pipeline_schedule=pipeline_schedule)
    method: alpa.parallel_method.PipeshardParallel
    # The operator clustering for unet is not sufficient
    method.layer_option = ManualLayerOption(remat_layer=True)

    use_grad_acc = benchmark_case.num_micro_batches > 1
    grad_func = alpa.grad if use_grad_acc else jax.grad
    state, batch, learning_rate_fn = prepare_unet_input_and_model(
        benchmark_case)
    train_step = get_train_step(learning_rate_fn,
                                False,
                                None,
                                method,
                                grad_func=grad_func)

    (latencies, max_mem_allocated, compilation_times,
     executable) = compile_and_benchmark_pipeshard_training_executable(
         benchmark_case.parallel_mode,
         niter,
         train_step,
         state, (batch,),
         profile_driver_time=profile_driver_time)

    # Profile submesh executables
    # del state
    # del metrics
    # for i, profiled in enumerate(executable.profile_all_executables()):
    #     pstr = f"Mesh {i}: "
    #     for k in profiled:
    #         pstr += f"Exec {k}: {profiled[k][0]}s; "
    #     print(pstr)
    # executable.dump_debug_info("tmp")

    # Compute statistics
    num_gpus = virtual_mesh.num_devices
    tflops = executable.flop_count / num_gpus / np.mean(latencies) / 1e12
    parameter_count = compute_param_number(state.params)

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

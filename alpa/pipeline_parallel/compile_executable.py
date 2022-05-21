"""Compile executables for pipeshard parallelism."""
import dataclasses 
import logging
import time
from typing import Callable, Sequence

from jax import linear_util as lu
from jax.core import gensym, AbstractValue
from jax.tree_util import PyTreeDef

from alpa.device_mesh import VirtualPhysicalMesh
from alpa.global_env import global_config
from alpa.pipeline_parallel.pipeshard_executable import PipeshardDriverExecutable
from alpa.pipeline_parallel.schedules import (GpipeSchedule, PipeDreamFlush,
                                              InferenceSchedule)
from alpa.pipeline_parallel.computation import (
    create_donation_mapping, generate_computations_from_protos,
    generate_sharded_xla_computations,
    generate_sharded_xla_computations_arguments, get_donatable_intermediate,
    mark_missing_vars_in_backward_computation_pipeline_marks, offload_remat,
    pipeline_dce, slice_closed_jaxpr_by_full_pipeline_marks,
    split_donate_invars, XlaShardedPipelineComputation)
from alpa.pipeline_parallel.apply_grad import (
    compute_grad_to_accumulate_grad,
    process_apply_gradient,
    split_compute_grad_and_apply_grad)
from alpa.pipeline_parallel.stage_construction import (
    cluster_layers_and_slice_mesh, StageOption)
from alpa.pipeline_parallel.stage_profiling import CompileWorkerPool
from alpa.shard_parallel.auto_sharding import AutoShardingOption
from alpa.util import get_var_mapping, trace_jaxpr_with_micro_batch, OrderedSet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compile_pipeshard_executable(fun: lu.WrappedFun,
                                 in_tree: PyTreeDef,
                                 out_tree_thunk: Callable[[], PyTreeDef],
                                 donated_invars: Sequence[bool],
                                 batch_invars: Sequence[bool],
                                 virtual_mesh: VirtualPhysicalMesh,
                                 num_microbatch: int,
                                 pipeline_schedule: str,
                                 default_as_option: AutoShardingOption,
                                 stage_option: StageOption,
                                 *avals: Sequence[AbstractValue]):
    """
    Compile a callable for pipeshard parallel which combines
    pipeline parallelism and 2d shard parallelsim.
    """
    debug_compilation_time(None)

    # Trace the function to get the jaxpr
    closed_jaxpr, _, batch_size = trace_jaxpr_with_micro_batch(
        fun, batch_invars, num_microbatch, avals)
    gensym_func = gensym([closed_jaxpr.jaxpr])
    debug_compilation_time("trace")

    # Split the jaxpr into compute_grad and apply_grad
    (closed_jaxpr, compute_grad_jaxpr, apply_grad_jaxpr,
     microbatch_bound) = split_compute_grad_and_apply_grad(
         closed_jaxpr, gensym_func, num_microbatch)
    # FIXME(yonghao): use apply grad jaxpr returned by this function
    batch_dim = 0
    (reduction_vector, post_microbatch_bound,
     _apply_grad_jaxpr) = _get_full_batch_apply_grad(fun, avals, batch_invars,
                                                     microbatch_bound,
                                                     num_microbatch, batch_dim)

    if num_microbatch > 1:
        (acc_grad_jaxpr, acc_grad_dict,
         grad_in_to_out) = compute_grad_to_accumulate_grad(
             compute_grad_jaxpr, reduction_vector, gensym_func)
    else:
        acc_grad_jaxpr = compute_grad_jaxpr
        acc_grad_dict = {x: x for x in compute_grad_jaxpr.jaxpr.outvars}
        grad_in_to_out = {}

    # Slice the jaxpr into layers
    acc_grad_invars = acc_grad_jaxpr.jaxpr.invars
    acc_grad_outvars = acc_grad_jaxpr.jaxpr.outvars

    jax_pipeline_layers = slice_closed_jaxpr_by_full_pipeline_marks(
        acc_grad_jaxpr)
    assert (len(jax_pipeline_layers) == len(
        set(layer.name for layer in jax_pipeline_layers))), \
        "All layers must have unique names."
    jax_pipeline_layers = mark_missing_vars_in_backward_computation_pipeline_marks(
        jax_pipeline_layers, acc_grad_invars, acc_grad_outvars, gensym_func)
    # TODO(yonghao): remove this pass. we can clear these vars when rewriting
    # compute grad to accumulate grad
    jax_pipeline_layers = pipeline_dce(jax_pipeline_layers, acc_grad_outvars)
    jax_pipeline_layers = offload_remat(jax_pipeline_layers, gensym_func)

    # Initialize donation map
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    donation_mapping = dict(grad_in_to_out)

    inference_mode = (pipeline_schedule == "inference")
    (jax_apply_layers,
     apply_grad_global_info) = _slice_apply_grad_for_stage_construction(
         jax_pipeline_layers, apply_grad_jaxpr, microbatch_bound, acc_grad_dict,
         global_invars, global_outvars, donated_invars, donation_mapping,
         reduction_vector, num_microbatch, gensym_func, inference_mode)
    debug_compilation_time("jaxpr operations")

    # Construct pipeline stages by merging layers
    (jax_pipeline_stages, stage_to_mesh, sliced_virtual_meshes,
     logical_mesh_shapes, autosharding_option_dicts) = cluster_layers_and_slice_mesh(
         jax_pipeline_layers, virtual_mesh, donation_mapping,
         acc_grad_outvars, num_microbatch, batch_size,
         jax_apply_layers, apply_grad_global_info, pipeline_schedule,
         default_as_option, stage_option)
    num_meshes = len(sliced_virtual_meshes)
    debug_compilation_time("stage construction")

    # Process apply_gradient and donation
    (sliced_apply_grad_stages, n_stages, dependency, apply_grad_placement,
     global_outvars, donated_invars) = process_apply_gradient(
         apply_grad_jaxpr, microbatch_bound, acc_grad_dict, jax_pipeline_stages,
         stage_to_mesh, gensym_func, num_microbatch, num_meshes, global_invars,
         global_outvars, donated_invars, reduction_vector)
    jax_all_stages = jax_pipeline_stages + sliced_apply_grad_stages

    donation_mapping = create_donation_mapping(donation_mapping, donated_invars,
                                               global_invars, global_outvars)
    donate_invars_dict, jax_all_stages = split_donate_invars(
        donation_mapping, jax_all_stages, gensym_func)
    debug_compilation_time("apply grad")

    # Generate pipeline schedule and placement
    if pipeline_schedule == "gpipe":
        schedule = GpipeSchedule(dependency=dependency,
                                 meshes=sliced_virtual_meshes,
                                 apply_grad_placement=apply_grad_placement,
                                 num_batch=num_microbatch)
    elif pipeline_schedule == "1f1b":
        schedule = PipeDreamFlush(dependency=dependency,
                                  meshes=sliced_virtual_meshes,
                                  apply_grad_placement=apply_grad_placement,
                                  num_batch=num_microbatch)
    elif pipeline_schedule == "inference":
        schedule = InferenceSchedule(dependency=dependency,
                                     meshes=sliced_virtual_meshes,
                                     apply_grad_placement=apply_grad_placement,
                                     num_batch=num_microbatch)
    else:
        raise ValueError(f"Invalid schedule: {pipeline_schedule}")

    # Call auto-sharding pass to shard each stage
    xla_stages, total_flops = shard_each_stage(
        jax_all_stages, sliced_virtual_meshes, schedule, n_stages, num_meshes,
        grad_in_to_out, global_invars, acc_grad_outvars, donate_invars_dict,
        num_microbatch, logical_mesh_shapes, autosharding_option_dicts,
        default_as_option, gensym_func)
    total_flops *= num_microbatch
    debug_compilation_time("shard stages")

    # Launch the physical mesh group
    if virtual_mesh.launched_physical_mesh_group is None:
        virtual_mesh.get_physical_mesh_group(sliced_virtual_meshes)
    debug_compilation_time("launch meshes")

    # Wrap all things into a distributed runtime
    global_outvars, concat_vars_mapping = _rewrite_global_outvars_post_concate(
        global_outvars, reduction_vector, microbatch_bound,
        post_microbatch_bound, gensym_func)

    executable = PipeshardDriverExecutable(
        stages=xla_stages,
        global_invars=global_invars,
        grad_dummy_invars=grad_in_to_out,
        global_outvars=global_outvars,
        mesh_group=virtual_mesh.launched_physical_mesh_group,
        dependency=dependency,
        schedule=schedule,
        is_batch=batch_invars,
        num_batch=num_microbatch,
        flop_count=total_flops,
        concat_vars_mapping=concat_vars_mapping,
        in_tree=in_tree)
    debug_compilation_time("driver executable")
    return executable


def shard_each_stage(jax_all_stages, virtual_meshes, schedule, n_stages,
                     num_meshes, grad_in_to_out, global_invars,
                     acc_grad_outvars, donate_invars_dict, num_microbatch,
                     logical_mesh_shapes, autosharding_option_dicts,
                     default_as_option, gensym_func):
    """Run intra-op parallelism compilation for a stage."""
    # Initialize donation mapping
    stage_dict = [[] for _ in range(num_meshes)]
    stage_id_dict = [[] for _ in range(num_meshes)]
    dummy_stage_id_dict = [[] for _ in range(num_meshes)]
    donatable_dict = [[] for _ in range(num_meshes)]
    mesh_stage_mapping = schedule.mesh_stage_mapping
    donatable_list = get_donatable_intermediate(
        jax_all_stages, mesh_stage_mapping,
        OrderedSet(global_invars).union(grad_in_to_out.keys()))

    for i, stage in enumerate(jax_all_stages):
        mesh_indices = list(schedule.stage_placement(i))
        assert len(mesh_indices) == 1
        mesh_idx = mesh_indices[0]
        if len(stage.outvars) == 0:
            # This is a dummy stage, we don't need to shard it
            dummy_stage_id_dict[mesh_idx].append(i)
            continue
        stage_id_dict[mesh_idx].append(i)
        stage_dict[mesh_idx].append(stage)
        donatable_dict[mesh_idx].append(donatable_list[i])

    # Call auto-sharding pass on each stage
    distributed_compile = global_config.pipeline_distributed_compile
    xla_stages = [None] * n_stages
    if distributed_compile:
        compile_workers = CompileWorkerPool(num_meshes)
        compile_fn = lambda w, v: w.run_auto_sharding_pass.remote(*v)  # noqa
        compile_intermediate = [None] * num_meshes
    total_flops = 0
    for mesh_idx in range(num_meshes):
        virtual_mesh = virtual_meshes[mesh_idx]
        logical_mesh = virtual_mesh.get_logical_mesh(
            logical_mesh_shapes[mesh_idx])
        autosharding_option = dataclasses.replace(
            default_as_option, **autosharding_option_dicts[mesh_idx])

        # Setup dummy stages
        for i in dummy_stage_id_dict[mesh_idx]:
            xla_stages[i] = XlaShardedPipelineComputation.dummy_computation(
                jax_all_stages[i].name, logical_mesh.shape, gensym_func)

        stage_donate_invars = [
            donate_invars_dict[stage_idx]
            for stage_idx in stage_id_dict[mesh_idx]
        ]
        if distributed_compile:
            proto, jaxpr_args, flops = generate_sharded_xla_computations_arguments(
                str(mesh_idx), stage_dict[mesh_idx], stage_donate_invars)
            other_kwargs = {
                "logical_mesh": logical_mesh,
                "return_mode": "stage_protos",
                "as_option": autosharding_option,
                "num_micro_batches": num_microbatch,
            }
            compile_workers.submit(compile_fn,
                                   (mesh_idx, proto, jaxpr_args, other_kwargs))
            compile_intermediate[mesh_idx] = (stage_dict[mesh_idx],
                                              stage_donate_invars)
            total_flops += flops
        else:
            sharded_xla_stages, flops = generate_sharded_xla_computations(
                str(mesh_idx), stage_dict[mesh_idx], stage_donate_invars,
                donatable_dict[mesh_idx], acc_grad_outvars, num_microbatch,
                logical_mesh, autosharding_option)
            total_flops += flops
            for i, xla_stage in zip(stage_id_dict[mesh_idx],
                                    sharded_xla_stages):
                xla_stages[i] = xla_stage

    if distributed_compile:
        for _ in range(num_meshes):
            mesh_idx, (computation_names, computation_protos,
                       strategy_config) = compile_workers.get_next_unordered()
            jax_computations, computation_donate_invars = compile_intermediate[
                mesh_idx]
            sharded_xla_stages = generate_computations_from_protos(
                jax_computations, computation_names, computation_protos,
                computation_donate_invars, donatable_dict[mesh_idx],
                acc_grad_outvars, strategy_config)
            for i, xla_stage in zip(stage_id_dict[mesh_idx],
                                    sharded_xla_stages):
                xla_stages[i] = xla_stage
        compile_workers.shutdown()

    return xla_stages, total_flops


def _slice_apply_grad_for_stage_construction(pipeline_layers, apply_grad_jaxpr,
                                             microbatch_bound, acc_grad_dict,
                                             global_invars, global_outvars,
                                             donated_invars, donation_mapping,
                                             reduction_vector, num_microbatch,
                                             gensym_func, inference_mode):
    if inference_mode:
        num_layers = len(pipeline_layers)
        num_mesh = num_layers
        layer_to_mesh = list(range(num_mesh))
    else:
        num_layers = len(pipeline_layers)
        assert len(pipeline_layers) % 2 == 0
        num_mesh = num_layers // 2
        layer_to_mesh = (list(range(num_mesh)) +
                         list(reversed(range(num_mesh))))
    (layers, _, _,
     apply_grad_placement, _, donated_invars) = process_apply_gradient(
         apply_grad_jaxpr, microbatch_bound, acc_grad_dict, pipeline_layers,
         layer_to_mesh, gensym_func, num_microbatch, num_mesh, global_invars,
         global_outvars, donated_invars, reduction_vector)
    apply_grad_donation = create_donation_mapping(donation_mapping,
                                                  donated_invars, global_invars,
                                                  global_outvars)
    wrap_layers = [None] * num_mesh
    for layer_idx, mesh_idx in apply_grad_placement.items():
        wrap_layers[mesh_idx] = layers[layer_idx - num_layers]
    apply_grad_global_info = apply_grad_donation, global_outvars
    return wrap_layers, apply_grad_global_info


# TODO(yonghao): the reduction vector should be created by a more careful analysis.
def _get_full_batch_apply_grad(fun: lu.WrappedFun, avals, batch_invars,
                               microbatch_bound, num_microbatch, batch_dim):
    # Trace and split a non-microbatch version for the correct shape of
    # Global output and apply grad
    dummy_microbatch = 1
    stores = [lu.Store() for _ in fun.stores]
    clone = lu.WrappedFun(fun.f, fun.transforms, stores, fun.params)
    closed_jaxpr, _, _ = trace_jaxpr_with_micro_batch(clone, batch_invars,
                                                      dummy_microbatch, avals)
    gensym_func = gensym([closed_jaxpr.jaxpr])
    (closed_jaxpr, _, apply_grad_jaxpr,
     dummy_microbatch_bound) = (split_compute_grad_and_apply_grad(
         closed_jaxpr, gensym_func, num_microbatch))
    reduced_vector = []
    for mb_var, var in zip(microbatch_bound.outvars,
                           dummy_microbatch_bound.outvars):
        microbatch_shape = mb_var.aval.shape
        batch_shape = var.aval.shape
        if microbatch_shape != batch_shape:
            expected_microbatched_shape = list(batch_shape)
            assert expected_microbatched_shape[batch_dim] % num_microbatch == 0
            expected_microbatched_shape[batch_dim] //= num_microbatch
            assert tuple(expected_microbatched_shape) == microbatch_shape
            if len(apply_grad_jaxpr.eqns) > 0:
                raise NotImplementedError(
                    "apply gradient with not reduced input is not supported yet."
                )
        reduced_vector.append(microbatch_shape == batch_shape)

    return reduced_vector, dummy_microbatch_bound, apply_grad_jaxpr


def _rewrite_global_outvars_post_concate(global_outvars, reduction_vector,
                                         microbatch_bound,
                                         post_microbatch_bound, gensym_func):
    concat_vars_mapping = {}
    for idx, reduce in enumerate(reduction_vector):
        if not reduce:
            var = microbatch_bound.outvars[idx]
            actual_aval = post_microbatch_bound.outvars[idx].aval
            concat_vars_mapping[gensym_func(actual_aval)] = var
    reversed_mapping = {v: k for k, v in concat_vars_mapping.items()}
    global_outvars = [
        get_var_mapping(reversed_mapping, v) for v in global_outvars
    ]
    return global_outvars, concat_vars_mapping


_tic = None


def debug_compilation_time(message):
    """Print compilation time for debugging."""
    global _tic
    if message and global_config.print_compilation_time:
        print(f"compile_pipeshard_executable::{message}: "
              f"{time.time() - _tic:.2f} s")
    _tic = time.time()

"""Compile executables for pipeshard parallelism."""
import dataclasses
import logging
import time
from typing import Callable, Sequence, Optional

from jax import linear_util as lu
from jax._src.lib import xla_client as xc
from jax.core import gensym, AbstractValue, ClosedJaxpr
from jax.interpreters import pxla
from jax.tree_util import PyTreeDef

from alpa.device_mesh import VirtualPhysicalMesh
from alpa.global_env import global_config
from alpa.pipeline_parallel.pipeshard_executable import PipeshardDriverExecutable
from alpa.pipeline_parallel.runtime_emitter import (
    OverlapFriendlyPipelineInstEmitter, PipelineInstEmitter)
from alpa.pipeline_parallel.schedules import (GpipeSchedule,
                                              OverlapFriendlyPipeDreamSchedule,
                                              PipeDreamFlush, InferenceSchedule)
from alpa.pipeline_parallel.computation import (
    create_donation_mapping, generate_computations_from_modules,
    generate_sharded_xla_computations,
    generate_sharded_xla_computations_arguments, get_donatable_intermediate,
    mark_missing_vars_in_backward_computation_pipeline_marks, pipeline_dce,
    slice_closed_jaxpr_by_full_pipeline_marks, split_donate_invars,
    XlaShardedPipelineComputation)
from alpa.pipeline_parallel.apply_grad import (
    apply_grad_get_mean, compute_grad_to_accumulate_grad,
    process_apply_gradient, split_compute_grad_and_apply_grad)
from alpa.pipeline_parallel.layer_construction import LayerOption
from alpa.pipeline_parallel.schedules import gen_dependency_with_stages
from alpa.pipeline_parallel.stage_construction import (
    cluster_layers_and_slice_mesh, StageOption)
from alpa.pipeline_parallel.stage_profiling import CompileWorkerPool
from alpa.shard_parallel.auto_sharding import (AutoShardingOption,
                                               hlo_sharding_to_sharding_spec)
from alpa.shard_parallel.manual_sharding import (ManualShardingOption,
                                                 ParsedManualShardingOption,
                                                 get_flatten_axis_resources,
                                                 parsed_spec_to_opsharding)
from alpa.util import (get_var_mapping, trace_jaxpr_with_micro_batch,
                       OrderedSet, GradFuncTransformContext)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compile_pipeshard_executable(
        fun: lu.WrappedFun, in_tree: PyTreeDef,
        out_tree_thunk: Callable[[], PyTreeDef], static_argnums: Sequence[int],
        donated_invars: Sequence[bool], batch_invars: Sequence[bool],
        virtual_mesh: VirtualPhysicalMesh, num_microbatch: int,
        pipeline_schedule: str, default_as_option: AutoShardingOption,
        layer_option: LayerOption, stage_option: StageOption,
        global_input_shardings: Optional[Sequence[pxla.ShardingSpec]],
        stage_input_shardings: Optional[Sequence[Sequence[pxla.ShardingSpec]]],
        manual_shard_options: Optional[ManualShardingOption],
        *avals: Sequence[AbstractValue]):
    """
    Compile a callable for pipeshard parallel which combines
    pipeline parallelism and 2d shard parallelsim.

    Args:
        fun: The function to be parallelized.
        global_input_shardings: Forcibly set sharding specs of global
          input vars.
        stage_input_shardings: Forcibly set sharding specs of input vars of
          each stage.
        manual_sharding_options: pjit style sharding constraints of global input
          vars.
    """
    if global_config.backend == "tpu":
        raise NotImplementedError("Pipeshard Parallel for tpu is not supported")
    debug_compilation_time(None)
    name_base = f"{fun.__name__}_pipeshard_parallel"

    # Apply layer construction to add pipeline markers.
    with GradFuncTransformContext(layer_option.transform):
        if pipeline_schedule == "inference":
            f_backup = fun.f
            fun.f = layer_option.transform(fun.f)

        # Trace the function with a micro batch to get the jaxpr.
        closed_jaxpr, micro_batch_size = trace_jaxpr_with_micro_batch(
            fun, batch_invars, num_microbatch, avals)

        # Trace again with a full batch.
        # The full batch is used to derive the reduction operator across
        # micro batches (e.g., addition, concatenation).
        if num_microbatch > 1:
            for store in fun.stores:
                if store:
                    store.reset()
            full_batch_closed_jaxpr, _ = trace_jaxpr_with_micro_batch(
                fun, batch_invars, 1, avals)
        else:
            full_batch_closed_jaxpr = None

        if pipeline_schedule == "inference":
            fun.f = f_backup
    debug_compilation_time("trace")

    # flatten manual sharding axis resources
    out_tree = out_tree_thunk()
    if manual_shard_options is not None:
        assert global_input_shardings is None
        parsed_ms_option = get_flatten_axis_resources(manual_shard_options,
                                                      in_tree, out_tree)
    else:
        parsed_ms_option = None
    pipeshard_config = compile_pipeshard_executable_internal(
        closed_jaxpr, full_batch_closed_jaxpr, micro_batch_size, donated_invars,
        batch_invars, virtual_mesh, num_microbatch, pipeline_schedule,
        default_as_option, stage_option, name_base, global_input_shardings,
        None, stage_input_shardings, parsed_ms_option)

    executable = PipeshardDriverExecutable(
        mesh_group=virtual_mesh.launched_physical_mesh_group,
        pipeshard_config=pipeshard_config,
        num_batch=num_microbatch,
        layer_option=layer_option,
        in_tree=in_tree,
        out_tree=out_tree,
        static_argnums=static_argnums)
    debug_compilation_time("driver executable")
    return executable


def compile_pipeshard_executable_internal(
        closed_jaxpr: ClosedJaxpr,
        full_batch_closed_jaxpr: Optional[ClosedJaxpr], micro_batch_size: int,
        donated_invars: Sequence[bool], batch_invars: Sequence[bool],
        virtual_mesh: VirtualPhysicalMesh, num_microbatch: int,
        pipeline_schedule: str, default_as_option: AutoShardingOption,
        stage_option: StageOption, name_base: str,
        global_input_shardings: Optional[Sequence[pxla.ShardingSpec]],
        global_output_shardings: Optional[Sequence[pxla.ShardingSpec]],
        stage_input_shardings: Optional[Sequence[Sequence[pxla.ShardingSpec]]],
        parsed_manual_sharding_option: Optional[ParsedManualShardingOption]):
    """
    Args:
        fun: The function to be parallelized.
        global_input_shardings: Forcibly set sharding specs of global
          input vars.
        global_output_shardings: Forcibly set sharding specs of global
          output vars.
        stage_input_shardings: Forcibly set sharding specs of input vars of
          each stage.
    """
    global_invars = closed_jaxpr.jaxpr.invars
    gensym_func = gensym([closed_jaxpr.jaxpr])
    inference_mode = (pipeline_schedule == "inference")

    (closed_jaxpr, global_outvars, jax_pipeline_layers, apply_grad_jaxpr,
     microbatch_bound, reduction_vector, post_microbatch_bound,
     accumulator_mapping, acc_grad_invars,
     acc_grad_outvars) = (split_and_process_layers(closed_jaxpr,
                                                   full_batch_closed_jaxpr,
                                                   num_microbatch,
                                                   inference_mode, gensym_func))

    debug_compilation_time("jaxpr operations")

    (jax_apply_layers,
     apply_grad_global_info) = slice_apply_grad_for_stage_construction(
         jax_pipeline_layers, apply_grad_jaxpr, microbatch_bound, global_invars,
         global_outvars, donated_invars, accumulator_mapping, gensym_func,
         inference_mode)

    # Construct pipeline stages by merging layers
    (jax_pipeline_stages, stage_to_mesh, sliced_virtual_meshes,
     manual_stage_option) = cluster_layers_and_slice_mesh(
         jax_pipeline_layers, virtual_mesh, accumulator_mapping,
         acc_grad_invars, acc_grad_outvars, num_microbatch, micro_batch_size,
         jax_apply_layers, apply_grad_global_info, pipeline_schedule,
         default_as_option, stage_option)
    num_meshes = len(sliced_virtual_meshes)
    debug_compilation_time("stage construction")

    # Process apply_gradient and donation
    num_devices = [vmesh.num_devices for vmesh in sliced_virtual_meshes]
    (sliced_apply_grad_stages, apply_grad_placement,
     global_outvars, allreduce_groups) = process_apply_gradient(
         apply_grad_jaxpr, microbatch_bound, jax_pipeline_stages, stage_to_mesh,
         gensym_func, num_meshes, global_invars, global_outvars, donated_invars,
         False, num_devices)
    jax_all_stages = jax_pipeline_stages + sliced_apply_grad_stages

    donation_mapping = create_donation_mapping(accumulator_mapping,
                                               donated_invars, global_invars,
                                               global_outvars)
    donate_invars_dict, jax_all_stages = split_donate_invars(
        donation_mapping, jax_all_stages, gensym_func)
    global_outvars, concat_vars_mapping = _rewrite_global_outvars_post_concate(
        global_outvars, reduction_vector, microbatch_bound,
        post_microbatch_bound, gensym_func)
    debug_compilation_time("apply grad")

    # Generate pipeline schedule and placement
    dependency = gen_dependency_with_stages(jax_pipeline_stages,
                                            sliced_apply_grad_stages)
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
    elif pipeline_schedule == "1f1b_overlap_friendly":
        schedule = OverlapFriendlyPipeDreamSchedule(
            dependency=dependency,
            meshes=sliced_virtual_meshes,
            apply_grad_placement=apply_grad_placement,
            num_batch=num_microbatch)
    else:
        raise ValueError(f"Invalid schedule: {pipeline_schedule}")

    # Forcibly set the sharding specs of global invars and outvars.
    # FIXME(yonghao): the invar can appear on multiple meshes and thus different
    # sharding specs
    if global_input_shardings:
        assert len(global_input_shardings) == len(global_invars)
        input_sharding_dict = dict(zip(global_invars, global_input_shardings))
    else:
        input_sharding_dict = {}
    if global_output_shardings:
        assert len(global_output_shardings) == len(global_outvars)
        output_sharding_dict = dict(zip(global_outvars,
                                        global_output_shardings))
    else:
        output_sharding_dict = {}
    if parsed_manual_sharding_option is not None:
        assert (global_input_shardings is None and
                global_output_shardings is None)
        (input_sharding_dicts,
         output_sharding_dicts) = get_manual_input_output_sharding_specs(
             jax_all_stages, manual_stage_option.submesh_logical_shapes,
             parsed_manual_sharding_option, global_invars, global_outvars,
             schedule.stage_mesh_mapping)
    else:
        input_sharding_dicts = [input_sharding_dict] * num_meshes
        output_sharding_dicts = [output_sharding_dict] * num_meshes

    # Call auto-sharding pass to shard each stage
    xla_stages, total_flops = shard_each_stage(
        jax_all_stages, sliced_virtual_meshes, schedule, num_meshes,
        accumulator_mapping, global_invars, acc_grad_outvars,
        donate_invars_dict, num_microbatch,
        manual_stage_option.submesh_logical_shapes,
        manual_stage_option.submesh_autosharding_option_dicts,
        default_as_option, input_sharding_dicts, output_sharding_dicts,
        stage_input_shardings, name_base, gensym_func)
    total_flops *= num_microbatch
    debug_compilation_time("shard stages")

    # Launch the physical mesh group
    if virtual_mesh.launched_physical_mesh_group is None:
        virtual_mesh.get_physical_mesh_group(sliced_virtual_meshes)
    debug_compilation_time("launch meshes")

    # Wrap all things into a distributed runtime
    # TODO(yonghao): use virtual mesh instead of launched physical group
    emitter_kwargs = dict(stages=xla_stages,
                          global_invars=global_invars,
                          grad_dummy_invars=accumulator_mapping,
                          global_outvars=global_outvars,
                          concat_vars_mapping=concat_vars_mapping,
                          mesh_group=virtual_mesh.launched_physical_mesh_group,
                          schedule=schedule,
                          is_batch=batch_invars,
                          num_batch=num_microbatch,
                          default_auto_sharding_option=default_as_option,
                          manual_stage_option=manual_stage_option,
                          flop_count=total_flops,
                          allreduce_groups=allreduce_groups)
    if pipeline_schedule == "1f1b_overlap_friendly":
        emitter_cls = OverlapFriendlyPipelineInstEmitter
        emitter_kwargs["outvar_def_order"] = [
            stage.outvars_def_order() for stage in jax_all_stages
        ]
    else:
        emitter_cls = PipelineInstEmitter
    pipeshard_config = emitter_cls(**emitter_kwargs).compile()

    debug_compilation_time("runtime emitter")
    return pipeshard_config


def split_and_process_layers(closed_jaxpr, full_batch_closed_jaxpr,
                             num_microbatch, inference_mode, gensym_func):
    """Split and process the input jaxpr with the following steps:

    1. Split the jaxpr into the compute grad part and the apply grad part.
    2. Transform the compute grad jaxpr to a accumulate grad jaxpr.
    3. Split the accumulate grad jaxpr into forward and backward pipeline
       layers.
    4. Divide the accumulated gradient by the number of microbatches at the
       start of accumulate gradient.

    """

    # Split the jaxpr into compute_grad and apply_grad
    (closed_jaxpr, compute_grad_jaxpr, apply_grad_jaxpr,
     microbatch_bound) = split_compute_grad_and_apply_grad(
         closed_jaxpr, gensym_func, num_microbatch, inference_mode)
    global_outvars = closed_jaxpr.jaxpr.outvars

    # Transform compute_grad to accumulate_grad
    # FIXME(yonghao): use apply grad jaxpr returned by this function
    (reduction_vector, post_microbatch_bound,
     _) = _get_full_batch_apply_grad(full_batch_closed_jaxpr, microbatch_bound,
                                     num_microbatch, inference_mode)
    (acc_grad_jaxpr, microbatch_bound,
     accumulator_mapping) = compute_grad_to_accumulate_grad(
         compute_grad_jaxpr, microbatch_bound, reduction_vector, gensym_func,
         num_microbatch)

    # Slice the jaxpr into layers
    acc_grad_invars = acc_grad_jaxpr.jaxpr.invars
    acc_grad_outvars = acc_grad_jaxpr.jaxpr.outvars

    jax_pipeline_layers = slice_closed_jaxpr_by_full_pipeline_marks(
        acc_grad_jaxpr)
    if not inference_mode:
        jax_pipeline_layers = (
            mark_missing_vars_in_backward_computation_pipeline_marks(
                jax_pipeline_layers, acc_grad_invars, acc_grad_outvars,
                gensym_func))
    # TODO(yonghao): remove this pass. we can clear these vars when rewriting
    #   compute grad to accumulate grad
    jax_pipeline_layers = pipeline_dce(jax_pipeline_layers, acc_grad_outvars)

    # Add compute mean and slice apply-grad stages
    # FIXME (zhuohan): get_mean only works when we use jax.mean to
    #                  calculate loss. It will fail if we use sum.
    apply_grad_jaxpr, global_outvars = apply_grad_get_mean(
        apply_grad_jaxpr, global_outvars, microbatch_bound.outvars, gensym_func,
        num_microbatch, reduction_vector)

    return (closed_jaxpr, global_outvars, jax_pipeline_layers, apply_grad_jaxpr,
            microbatch_bound, reduction_vector, post_microbatch_bound,
            accumulator_mapping, acc_grad_invars, acc_grad_outvars)


def get_manual_input_output_sharding_specs(stages, mesh_shapes, ms_option,
                                           global_invars, global_outvars,
                                           stage_to_mesh):
    """
    Split user assigned input and output PartitionSpec into sharding specs for
    each pipeline stage.
    """
    invar_set = set(global_invars)
    outvar_set = set(global_outvars)
    var_to_pspec = {}
    handle_invar = False
    handle_outvar = False
    if ms_option.in_parsed_pspec is not None:
        var_to_pspec.update(dict(zip(global_invars, ms_option.in_parsed_pspec)))
        handle_invar = True
    if ms_option.out_parsed_pspec is not None:
        var_to_pspec.update(
            dict(zip(global_outvars, ms_option.out_parsed_pspec)))
        handle_outvar = True
    submesh_axis_names = ms_option.submesh_axis_names
    if submesh_axis_names is None:
        submesh_axis_names = [ms_option.mesh_axis_names] * len(mesh_shapes)

    def get_vars_to_sharding_specs(variables, mesh_shape, mesh_axis_names):
        parsed_specs = [var_to_pspec[v] for v in variables]
        avals = [v.aval for v in variables]
        var_op_shardings = parsed_spec_to_opsharding(parsed_specs, avals,
                                                     mesh_shape,
                                                     mesh_axis_names)
        var_sharding_specs = [
            hlo_sharding_to_sharding_spec(xc.HloSharding.from_proto(ops), aval,
                                          mesh_shape)
            for ops, aval in zip(var_op_shardings, avals)
        ]
        return dict(zip(variables, var_sharding_specs))

    invar_shardings = [{}] * len(mesh_shapes)
    outvar_shardings = [{}] * len(mesh_shapes)
    for stage_idx, stage in enumerate(stages):
        mesh_idx = stage_to_mesh[stage_idx]
        assert len(mesh_idx) == 1
        mesh_idx = list(mesh_idx)[0]
        mesh_shape = mesh_shapes[mesh_idx]
        mesh_axis_names = submesh_axis_names[mesh_idx]
        # invars
        if handle_invar:
            invar_in_global = [var for var in stage.invars if var in invar_set]
            stage_invar_shardings = get_vars_to_sharding_specs(
                invar_in_global, mesh_shape, mesh_axis_names)
        else:
            stage_invar_shardings = {}
        # outvars
        if handle_outvar:
            outvar_in_global = [
                var for var in stage.outvars if var in outvar_set
            ]
            stage_outvar_shardings = get_vars_to_sharding_specs(
                outvar_in_global, mesh_shape, mesh_axis_names)
        else:
            stage_outvar_shardings = {}
        invar_shardings[mesh_idx].update(stage_invar_shardings)
        outvar_shardings[mesh_idx].update(stage_outvar_shardings)
    return invar_shardings, outvar_shardings


def shard_each_stage(jax_all_stages, virtual_meshes, schedule, num_meshes,
                     accumulator_mapping, global_invars, acc_grad_outvars,
                     donate_invars_dict, num_microbatch, logical_mesh_shapes,
                     autosharding_option_dicts, default_as_option,
                     input_sharding_dicts, output_sharding_dicts,
                     stage_input_shardings, name_base, gensym_func):
    """Run intra-op parallelism compilation for a stage."""
    # Initialize donation mapping
    stage_dict = [[] for _ in range(num_meshes)]
    stage_id_dict = [[] for _ in range(num_meshes)]
    dummy_stage_id_dict = [[] for _ in range(num_meshes)]
    donatable_dict = [[] for _ in range(num_meshes)]
    mesh_stage_mapping = schedule.mesh_stage_mapping
    donatable_list = get_donatable_intermediate(
        jax_all_stages, mesh_stage_mapping,
        OrderedSet(global_invars).union(accumulator_mapping.keys()))

    if stage_input_shardings is None:
        stage_input_shardings = [None for _ in range(num_meshes)]
    assert len(stage_input_shardings) == num_meshes

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
    xla_stages = [None] * len(jax_all_stages)
    if distributed_compile:
        compile_workers = CompileWorkerPool(num_meshes)
        compile_fn = lambda w, v: w.run_auto_sharding_pass.remote(*v)  # pylint: disable=unnecessary-lambda-assignment
        compile_intermediate = [None] * num_meshes
    total_flops = 0
    for mesh_idx in range(num_meshes):
        input_sharding_dict = input_sharding_dicts[mesh_idx]
        output_sharding_dict = output_sharding_dicts[mesh_idx]
        virtual_mesh = virtual_meshes[mesh_idx]
        logical_mesh = virtual_mesh.get_logical_mesh(
            logical_mesh_shapes[mesh_idx])
        autosharding_option = dataclasses.replace(
            default_as_option, **autosharding_option_dicts[mesh_idx])
        stage_input_sharding = stage_input_shardings[mesh_idx]

        # Setup dummy stages
        for i in dummy_stage_id_dict[mesh_idx]:
            xla_stages[i] = XlaShardedPipelineComputation.dummy_computation(
                jax_all_stages[i].name, logical_mesh.shape, gensym_func)

        stage_donate_invars = [
            donate_invars_dict[stage_idx]
            for stage_idx in stage_id_dict[mesh_idx]
        ]
        if distributed_compile:
            hlo, flops = (generate_sharded_xla_computations_arguments(
                f"{name_base}_mesh_{mesh_idx}", stage_dict[mesh_idx],
                stage_donate_invars, input_sharding_dict, output_sharding_dict,
                stage_input_sharding))
            other_kwargs = {
                "logical_mesh": logical_mesh,
                "return_mode": "stages",
                "as_option": autosharding_option,
                "num_micro_batches": num_microbatch,
            }
            compile_workers.submit(compile_fn, (mesh_idx, hlo, other_kwargs))
            compile_intermediate[mesh_idx] = (stage_dict[mesh_idx],
                                              stage_donate_invars)
            total_flops += flops
        else:
            sharded_xla_stages, flops = generate_sharded_xla_computations(
                f"{name_base}_mesh_{mesh_idx}", stage_dict[mesh_idx],
                stage_donate_invars, donatable_dict[mesh_idx], acc_grad_outvars,
                num_microbatch, logical_mesh, autosharding_option,
                input_sharding_dict, output_sharding_dict, stage_input_sharding)
            total_flops += flops
            for i, xla_stage in zip(stage_id_dict[mesh_idx],
                                    sharded_xla_stages):
                xla_stages[i] = xla_stage

    if distributed_compile:
        for _ in range(num_meshes):
            mesh_idx, (computation_names, computation_hlos,
                       stage_plan) = compile_workers.get_next_unordered()
            jax_computations, computation_donate_invars = compile_intermediate[
                mesh_idx]
            sharded_xla_stages = generate_computations_from_modules(
                jax_computations, computation_names, computation_hlos,
                computation_donate_invars, donatable_dict[mesh_idx],
                acc_grad_outvars, stage_plan)
            for i, xla_stage in zip(stage_id_dict[mesh_idx],
                                    sharded_xla_stages):
                xla_stages[i] = xla_stage
        compile_workers.shutdown()

    return xla_stages, total_flops


def slice_apply_grad_for_stage_construction(pipeline_layers, apply_grad_jaxpr,
                                            microbatch_bound, global_invars,
                                            global_outvars, donated_invars,
                                            accumulator_mapping, gensym_func,
                                            inference_mode):
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
    (layers, apply_grad_placement, global_outvars,
     _) = process_apply_gradient(apply_grad_jaxpr, microbatch_bound,
                                 pipeline_layers, layer_to_mesh, gensym_func,
                                 num_mesh, global_invars, global_outvars,
                                 donated_invars, True, None)
    apply_grad_donation = create_donation_mapping(accumulator_mapping,
                                                  donated_invars, global_invars,
                                                  global_outvars)
    wrap_layers = [None] * num_mesh
    for layer_idx, mesh_idx in apply_grad_placement.items():
        wrap_layers[mesh_idx] = layers[layer_idx - num_layers]
    apply_grad_global_info = apply_grad_donation, global_outvars
    return wrap_layers, apply_grad_global_info


def _get_full_batch_apply_grad(closed_jaxpr,
                               microbatch_bound,
                               num_microbatch,
                               inference_mode,
                               batch_dim=0):
    """
    Compare the micro-batch jaxpr and full-batch jaxpr. Return whether
    the out var's is reduced across micro-batches.

    TODO(yonghao): the reduction vector should be created by a
    more careful analysis.
    """
    if num_microbatch == 1:
        reduced_vector = [True] * len(microbatch_bound.outvars)
        post_microbatch_bound = microbatch_bound
        apply_grad_jaxpr = None
        return reduced_vector, post_microbatch_bound, apply_grad_jaxpr

    gensym_func = gensym([closed_jaxpr.jaxpr])
    (_, _, apply_grad_jaxpr,
     post_microbatch_bound) = (split_compute_grad_and_apply_grad(
         closed_jaxpr, gensym_func, num_microbatch, inference_mode))
    reduced_vector = []
    for mb_var, var in zip(microbatch_bound.outvars,
                           post_microbatch_bound.outvars):
        microbatch_shape = mb_var.aval.shape
        batch_shape = var.aval.shape
        if microbatch_shape != batch_shape:
            expected_microbatched_shape = list(batch_shape)
            assert expected_microbatched_shape[batch_dim] % num_microbatch == 0
            expected_microbatched_shape[batch_dim] //= num_microbatch
            assert tuple(expected_microbatched_shape) == microbatch_shape
            if len(apply_grad_jaxpr.eqns) > 0:
                raise NotImplementedError(
                    "Some vars marked by gradient markers are not reduced "
                    "but concatenated. This case in the training mode "
                    "is not supported yet.")
        reduced_vector.append(microbatch_shape == batch_shape)

    return reduced_vector, post_microbatch_bound, apply_grad_jaxpr


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

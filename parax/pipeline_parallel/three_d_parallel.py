"""Generate callables for 3d parallel that combines pipelining and 2d sharding."""
import logging

from jax import linear_util as lu
from jax.core import ClosedJaxpr, gensym

from parax.device_mesh import VirtualPhysicalMesh
from parax.global_env import global_config
from parax.pipeline_parallel.decentralized_distributed_runtime import DecentralizedDistributedRuntime
from parax.pipeline_parallel.local_pipeline_parallel import LocalRuntime
from parax.pipeline_parallel.schedules import (GpipeSchedule,
                                               gen_dependency_with_stages,
                                               PipeDreamFlush)
from parax.pipeline_parallel.computation import (
    create_donation_mapping, generate_computations_from_protos,
    generate_sharded_xla_computations,
    generate_sharded_xla_computations_arguments, get_donatable_intermediate,
    mark_missing_vars_in_backward_computation_pipeline_marks, offload_remat,
    pipeline_dce, slice_closed_jaxpr_by_full_pipeline_marks,
    split_donate_invars, XlaShardedPipelineComputation)
from parax.pipeline_parallel.apply_grad import (
    compute_grad_to_accumulate_grad, process_apply_gradient,
    split_compute_grad_and_apply_grad)
from parax.pipeline_parallel.stage_construction import cluster_layers_and_slice_mesh
from parax.pipeline_parallel.stage_profiling import CompileWorkerPool
from parax.shard_parallel.auto_sharding import AutoShardingOption
from parax.util import trace_jaxpr_with_micro_batch, OrderedSet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@lu.cache
def three_d_parallel_callable(fun: lu.WrappedFun, in_tree, out_tree_thunk,
                              donated_invars, batch_invars, devices,
                              memory_budget_per_device, *avals):
    """3d parallel combining pipelining and 2d sharding."""

    if not global_config.with_physical_mesh:
        assert not (
            global_config.pipeline_stage_mode == "auto_gpipe" and
            global_config.cache_compute_cost == None
        ), "no physical mesh, cannot do auto gpipe without cached cost"

    if not isinstance(devices, VirtualPhysicalMesh):
        raise RuntimeError("Unrecognized type of `devices`, got: {}, "
                           "expected type: {}.".format(type(devices),
                                                       "VirtualPhysicalMesh"))

    # Trace the function to get the jaxpr
    num_micro_batches = global_config.num_micro_batches
    if num_micro_batches is None:
        logger.warning("num microbatch is unset. Use 1 by default.")
        num_micro_batches = 1
    closed_jaxpr, _, batch_size = trace_jaxpr_with_micro_batch(
        fun, batch_invars, num_micro_batches, avals)

    # Split the jaxpr into compute_grad and apply_grad
    gensym_func = gensym([closed_jaxpr.jaxpr])
    compute_grad_jaxpr, apply_grad_jaxpr, barrier = (
        split_compute_grad_and_apply_grad(closed_jaxpr))
    have_apply_grad = barrier is not None

    if have_apply_grad:
        acc_grad_jaxpr, acc_grad_dict, grad_in_to_out = compute_grad_to_accumulate_grad(
            compute_grad_jaxpr, gensym_func)
    else:
        acc_grad_jaxpr = compute_grad_jaxpr
        acc_grad_dict = {}
        grad_in_to_out = {}

    # Slice the jaxpr into layers
    acc_grad_invars = acc_grad_jaxpr.jaxpr.invars
    acc_grad_outvars = acc_grad_jaxpr.jaxpr.outvars

    jax_pipeline_layers = slice_closed_jaxpr_by_full_pipeline_marks(
        acc_grad_jaxpr)
    jax_pipeline_layers = mark_missing_vars_in_backward_computation_pipeline_marks(
        jax_pipeline_layers, acc_grad_invars, acc_grad_outvars)
    jax_pipeline_layers = pipeline_dce(jax_pipeline_layers, acc_grad_outvars)
    offload_remat(jax_pipeline_layers, gensym_func)

    # Initialize donation map
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    if have_apply_grad:
        donation_mapping = dict(grad_in_to_out)
    else:
        donation_mapping = dict()

    num_forward_layers = len(jax_pipeline_layers) // 2
    layer_to_dummy_mesh = (list(range(num_forward_layers)) +
                           list(reversed(range(num_forward_layers))))
    # FIXME(yonghao): not consider the case that a pair of layers have no apply gradient part
    (jax_apply_layers, _, _, _, dummy_global_outvars,
     dummy_donated_invars) = process_apply_gradient(
         apply_grad_jaxpr, barrier, acc_grad_dict, jax_pipeline_layers,
         layer_to_dummy_mesh, gensym_func, num_micro_batches,
         len(jax_pipeline_layers) // 2, global_invars, global_outvars,
         donated_invars)
    apply_grad_donation = create_donation_mapping(donation_mapping,
                                                  dummy_donated_invars,
                                                  global_invars, global_outvars)
    apply_grad_global_info = apply_grad_donation, global_outvars

    # Construct pipeline stages by merging layers
    virtual_mesh = devices
    (jax_pipeline_stages, stage_to_mesh, sliced_meshes, logical_mesh_shapes,
     autosharding_option_dicts) = cluster_layers_and_slice_mesh(
         jax_pipeline_layers,
         virtual_mesh,
         donation_mapping,
         acc_grad_outvars,
         num_micro_batches,
         batch_size,
         jax_apply_layers=jax_apply_layers,
         apply_grad_global_info=apply_grad_global_info,
         pipeline_stage_mode=global_config.pipeline_stage_mode,
         logical_mesh_search_space=global_config.logical_mesh_search_space,
         cache_compute_cost=global_config.cache_compute_cost,
         forward_stage_layer_ids=global_config.forward_stage_layer_ids,
         submesh_shapes=global_config.sub_physical_mesh_shapes,
         logical_mesh_shapes=global_config.sub_logical_mesh_shapes,
         autosharding_option_dicts=global_config.
         submesh_autosharding_option_dicts)

    num_meshes = len(sliced_meshes)

    # Process apply_gradient and donation
    if have_apply_grad:
        (sliced_apply_grad_stages, n_stages, dependency, apply_grad_placement,
         global_outvars, donated_invars) = process_apply_gradient(
             apply_grad_jaxpr, barrier, acc_grad_dict, jax_pipeline_stages,
             stage_to_mesh, gensym_func, num_micro_batches, num_meshes,
             global_invars, global_outvars, donated_invars)
        jax_all_stages = jax_pipeline_stages + sliced_apply_grad_stages
    else:
        jax_all_stages = jax_pipeline_stages
        n_stages = len(jax_pipeline_stages)
        dependency = gen_dependency_with_stages(jax_pipeline_stages)
        apply_grad_placement = {}

    donation_mapping = create_donation_mapping(donation_mapping, donated_invars,
                                               global_invars, global_outvars)
    donate_invars_dict, jax_all_stages = split_donate_invars(
        donation_mapping, jax_all_stages)

    # Generate pipeline schedule and placement
    if global_config.pipeline_parallel_schedule == "gpipe":
        schedule = GpipeSchedule(dependency=dependency,
                                 meshes=sliced_meshes,
                                 apply_grad_placement=apply_grad_placement,
                                 num_batch=num_micro_batches)
    elif global_config.pipeline_parallel_schedule == "1f1b":
        schedule = PipeDreamFlush(dependency=dependency,
                                  meshes=sliced_meshes,
                                  apply_grad_placement=apply_grad_placement,
                                  num_batch=num_micro_batches)
    else:
        raise RuntimeError(
            "Unrecognized pipeline parallel schedule. "
            "Got `{}`. Availabe ones are `gpipe` or `1f1b`.".format(
                global_config.pipeline_parallel_schedule))
    if logger.level == logging.DEBUG:
        logger.debug(schedule.pprint_schedule(print=False))

    # Call auto-sharding pass to shard each stage
    xla_stages, total_flops = shard_each_stage(
        jax_all_stages, sliced_meshes, schedule, n_stages, num_meshes,
        grad_in_to_out, global_invars, acc_grad_outvars, donate_invars_dict,
        num_micro_batches, logical_mesh_shapes, autosharding_option_dicts,
        memory_budget_per_device, gensym_func)
    total_flops *= num_micro_batches

    # Debug use: only compile Hlo, even without enough device.
    if not global_config.with_physical_mesh:
        return LocalRuntime(pipeline_stages=xla_stages,
                            global_invars=global_invars,
                            global_outvars=global_outvars,
                            get_hlo_texts=True)

    # Wrap all things into a distributed runtime
    physical_meshes = [mesh.get_physical_mesh() for mesh in sliced_meshes]
    grad_in_to_out = {k: repr(v) for k, v in grad_in_to_out.items()}
    jp = DecentralizedDistributedRuntime(pipeline_stages=xla_stages,
                                         global_invars=global_invars,
                                         grad_dummy_invars=grad_in_to_out,
                                         global_outvars=global_outvars,
                                         physical_meshes=physical_meshes,
                                         dependency=dependency,
                                         schedule=schedule,
                                         is_batch=batch_invars,
                                         num_batch=num_micro_batches,
                                         flop_count=total_flops)

    def ret_func(*args, **kwargs):
        return jp.run(*args, **kwargs)

    ret_func.get_executable = lambda: jp
    return ret_func  # pylint: disable=unnecessary-lambda


def shard_each_stage(jax_all_stages, virtual_meshes, schedule, n_stages,
                     num_meshes, grad_in_to_out, global_invars,
                     acc_grad_outvars, donate_invars_dict, num_micro_batches,
                     logical_mesh_shapes, autosharding_option_dicts,
                     memory_budget_per_device, gensym_func):
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
    xla_stages = [None] * n_stages
    compile_workers = CompileWorkerPool(num_meshes, 1)
    compile_fn = lambda w, v: w.compile_proto_with_search.remote(*v)
    compile_intermediate = [None] * num_meshes
    total_flops = 0
    default_autosharding_option = global_config.default_autosharding_option
    for mesh_idx in range(num_meshes):
        virtual_mesh = virtual_meshes[mesh_idx]
        logical_mesh_choices = [
            virtual_mesh.get_logical_mesh(logical_mesh_shapes[mesh_idx])
        ]
        logical_mesh_search_mode = "cost_model"
        autosharding_option = default_autosharding_option.deepcopy_and_update(
            autosharding_option_dicts[mesh_idx])

        # Setup dummy stages
        for i in dummy_stage_id_dict[mesh_idx]:
            xla_stages[i] = XlaShardedPipelineComputation.dummy_computation(
                jax_all_stages[i].name, logical_mesh_choices[0].shape,
                gensym_func)

        stage_donate_invars = [
            donate_invars_dict[stage_idx]
            for stage_idx in stage_id_dict[mesh_idx]
        ]
        search_task = None
        record_file = None
        if global_config.pipeline_distributed_compile:
            proto, jaxpr_args, flops = generate_sharded_xla_computations_arguments(
                str(mesh_idx), stage_dict[mesh_idx], stage_donate_invars)
            mesh_kwargs = {
                "logical_mesh_choices": logical_mesh_choices,
                "return_mode": "stage_protos",
                "num_micro_batches": num_micro_batches,
                "bypass_device_assignment_check": True,
                "memory_budget_per_device": memory_budget_per_device,
                "logical_mesh_search_mode": logical_mesh_search_mode,
                "search_task": search_task,
                "record_file": record_file,
            }
            compile_workers.submit(
                compile_fn,
                (mesh_idx, proto, jaxpr_args, autosharding_option, mesh_kwargs))
            compile_intermediate[mesh_idx] = (stage_dict[mesh_idx],
                                              stage_donate_invars)
            total_flops += flops
        else:
            sharded_xla_stages, flops = generate_sharded_xla_computations(
                str(mesh_idx), stage_dict[mesh_idx], stage_donate_invars,
                donatable_dict[mesh_idx], acc_grad_outvars, num_micro_batches,
                logical_mesh_choices, autosharding_option,
                memory_budget_per_device, logical_mesh_search_mode, search_task,
                record_file)
            total_flops += flops
            for i, xla_stage in zip(stage_id_dict[mesh_idx],
                                    sharded_xla_stages):
                xla_stages[i] = xla_stage

    if global_config.pipeline_distributed_compile:
        for _ in range(num_meshes):
            mesh_idx, (computation_protos,
                       strategy_config) = compile_workers.get_next_unordered()
            jax_computations, computation_donate_invars = compile_intermediate[
                mesh_idx]
            sharded_xla_stages = generate_computations_from_protos(
                jax_computations, computation_protos, computation_donate_invars,
                donatable_dict[mesh_idx], acc_grad_outvars, strategy_config)
            for i, xla_stage in zip(stage_id_dict[mesh_idx],
                                    sharded_xla_stages):
                xla_stages[i] = xla_stage
    compile_workers.shutdown()

    return xla_stages, total_flops

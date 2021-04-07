"""Use the auto sharding pass in XLA"""
from functools import partial

import numpy as np

import jax
from jax import linear_util as lu
from jax.interpreters import xla, pxla, partial_eval as pe
from jax.lib import xla_bridge as xb
from jax.lib import xla_client as xc
from jax._src.util import (partial, unzip2, unzip3, prod, safe_map, safe_zip,
                           extend_name_stack, wrap_name, assert_unreachable,
                           tuple_insert, tuple_delete, curry)
from jaxlib.xla_client import OpSharding

xops = xc.ops

def auto_sharding_callable(
    fun: lu.WrappedFun,
    out_tree_thunk,
    devices,
    donated_invars,
    *avals
):
    devices = devices or np.array(jax.devices())

    # Trace to get jaxpr
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, avals)

    tuple_args = len(avals) > 100  # pass long arg lists as tuple for TPU

    # Make xla arguments
    c = xb.make_computation_builder(f"auto_shard_{fun.__name__}")
    xla_consts = map(partial(xb.constant, c), consts)
    xla_args, donated_invars = xla._xla_callable_args(c, avals, tuple_args, donated_invars=donated_invars)

    # Convert jaxpr to XLA HLO
    backend_name = 'gpu'
    axis_env = xla.AxisEnv(nreps=1, names=(), sizes=())  # All named axes have been vmapped
    transformed_name = fun.__name__
    out_nodes = xla.jaxpr_subcomp(
        c, jaxpr, backend_name, axis_env, xla_consts,
        extend_name_stack(wrap_name(transformed_name, 'auto_sharding')), *xla_args)
    out_tuple = xops.Tuple(c, out_nodes)

    # Set up aliases (donating invars)
    backend = xb.get_backend(backend_name)
    if backend.platform in ("gpu", "tpu"):
        donated_invars = xla.set_up_aliases(c, xla_args, out_tuple, donated_invars, tuple_args)
    if any(donated_invars):
        # TODO(tomhennigan): At call time we should mark these buffers as deleted.
        unused_donations = [str(c.GetShape(a))
                            for a, d in zip(xla_args, donated_invars) if d]
        warn("Some donated buffers were not usable: {}".format(", ".join(unused_donations)))

    # Compile
    device_ids = np.array([x.id for x in devices])
    num_replicas = 1
    num_partitions = len(device_ids)
    device_assignment = device_ids.reshape((num_replicas, num_partitions))
    spmd_lowering = True
    compile_options = xb.get_compile_options(
        num_replicas=num_replicas,
        num_partitions=num_partitions,
        device_assignment=device_assignment,
        use_spmd_partitioning=spmd_lowering,
    )
    compile_options.parameter_is_tupled_arguments = tuple_args
    built = c.Build(out_tuple)
    compiled = xla.backend_compile(backend, built, compile_options)

    # Handle args (re-shard if the layout is not the same)
    input_shardings = compiled.hlo_modules()[0].spmd_parameters_shardings()
    input_sharding_specs = [hlo_sharding_to_sharding_spec(proto_tuple, aval, num_partitions)
                           for (proto_tuple, aval) in zip(input_shardings, avals)]
    input_indices = [pxla.spec_to_indices(aval.shape, spec) for
                     aval, spec in zip(avals, input_sharding_specs)]
    handle_args = partial(pxla.shard_args, compiled.local_devices(), input_indices)

    #print("=" * 20)
    #print((compiled.hlo_modules()[0]).to_string())
    #print("=" * 20)

    # Handle output
    output_sharding = compiled.hlo_modules()[0].spmd_output_sharding()
    output_sharding_specs = hlo_sharding_to_sharding_spec(output_sharding, out_avals, num_partitions)
    handle_outs = pxla.avals_to_results_handler(num_replicas, num_partitions,
                                                output_sharding_specs, out_avals)

    return partial(pxla.execute_replicated, compiled, backend, handle_args, handle_outs)


def hlo_sharding_to_sharding_spec_no_tuple(proto_tuple, aval, num_partitions):
    sharding_type, tile_assignment_dimensions, tile_assignment_devices,\
        tuple_shardings, replicate_on_last_tile_dim = proto_tuple

    sharding = []
    mesh_mapping = []
    if sharding_type == OpSharding.Type.OTHER:
        for i in range(len(tile_assignment_dimensions)):
            sharding.append(pxla.Chunked([tile_assignment_dimensions[i]]))
            mesh_mapping.append(pxla.ShardedAxis(i))
    elif sharding_type == OpSharding.Type.REPLICATED:
        sharding = (pxla.NoSharding(),) * len(aval.shape)
        mesh_mapping = (pxla.Replicated(num_partitions),)
    else:
        raise NotImplementedError("Type: " + str(sharding_type))

    return pxla.ShardingSpec(sharding, mesh_mapping)


def hlo_sharding_to_sharding_spec(hlo_sharding, aval, num_partitions):
    proto_tuple = hlo_sharding.proto_tuple()
    sharding_type, tile_assignment_dimensions, tile_assignment_devices,\
        tuple_shardings, replicate_on_last_tile_dim = proto_tuple
    if sharding_type == OpSharding.Type.TUPLE:
        avals = aval
        return [hlo_sharding_to_sharding_spec_no_tuple(shard, aval, num_partitions)
                for (shard, aval) in zip(tuple_shardings, avals)]
    else:
        return hlo_sharding_to_sharding_spec_no_tuple(proto_tuple, aval, num_partitions)


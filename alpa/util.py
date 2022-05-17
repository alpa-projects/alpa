# pylint: disable=consider-using-enumerate
"""Common utilities."""
import functools
import itertools as it
import os
import subprocess
import time
from collections import OrderedDict
from datetime import datetime
from functools import partial, partialmethod
from typing import Sequence, Any
from warnings import warn

import jax
import jax.numpy as jnp
from jax._src import dispatch
from jax._src.api import FLAGS, ShapeDtypeStruct
from jax._src.dlpack import from_dlpack, to_dlpack
from jax._src.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
from jax.api_util import shaped_abstractify
from jax.core import (Atom, ClosedJaxpr, DropVar, Jaxpr, JaxprEqn, Literal,
                      ShapedArray, Var)
from jax.experimental.maps import FrozenDict
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla, pxla
from jax.interpreters.xla import _DeviceArray
from jax.tree_util import tree_map, tree_flatten, PyTreeDef
import numpy as np
import flax
from flax.training import train_state
import ray
import tqdm
import cupy as cp

from alpa.global_env import global_config, is_worker


########################################
##### Alpa API Utilities
########################################


def freeze_dict(pytree: PyTreeDef):
    """Convert a pytree to a FrozenDict."""

    def is_leaf(x):
        return isinstance(x, dict)

    def freeze(x):
        if isinstance(x, dict):
            return FrozenDict(x)
        return x

    return tree_map(freeze, pytree, is_leaf)


def auto_static_argnums(args: Sequence[Any]):
    """Return the indices of static arguments according to heuristic rules."""

    def is_static_arg(arg):
        if isinstance(arg, (bool, int, float, str)):
            return True

        if isinstance(arg, (flax.optim.base.Optimizer, train_state.TrainState)):
            return False

        xs, _ = tree_flatten(arg)
        for x in xs:
            try:
                x = shaped_abstractify(x)
            except TypeError:
                return True
        return False

    return tuple(i for i in range(len(args)) if is_static_arg(args[i]))


def auto_donate_argnums(args: Sequence[Any]):
    """Return the indices of donated arguments according to heuristic rules."""

    def should_donate(x):
        # Always donate optimizer
        if isinstance(x, (flax.optim.base.Optimizer, train_state.TrainState)):
            return True
        return False

    return tuple(i for i in range(len(args)) if should_donate(args[i]))


def abstractify_with_aval(x):
    if isinstance(x, ShapedArray):
        return x
    elif isinstance(x, ShapeDtypeStruct):
        return ShapedArray(x.shape, x.dtype, named_shape=x.named_shape)
    else:
        return xla.abstractify(x)


def tree_to_nparray(tree):
    """Convert a pytree to a pytree of numpy array."""

    def convert_to_nparray(x):
        if hasattr(x, "__array__"):
            return np.asanyarray(x)
        return x

    return tree_map(convert_to_nparray, tree)


def update_jax_platform(platform):
    """Update the jax backend platform."""
    jax.config.update("jax_platform_name", platform)
    xb.get_backend.cache_clear()


########################################
##### Data Structure Utilities
########################################


def to_int_tuple(array: np.ndarray):
    """Convert a numpy array to int tuple."""
    if array is None:
        return tuple()
    return tuple(int(x) for x in array)


def check_arithmetic_sequence(array: np.ndarray):
    """Check the input 1-D array is an arithmetic sequence. Return
    the delta if Ture and None otherwise."""
    if len(array) < 2:
        return None
    delta = array[1] - array[0]
    for i in range(2, len(array)):
        if array[i] - array[i - 1] != delta:
            return None
    return delta


class OrderedSet:
    """An ordered set implemented by using the built-in OrderedDict."""

    def __init__(self, iterable=()):
        self.dict = OrderedDict()
        for element in iterable:
            self.dict[element] = None

    def add(self, *args):
        for x in args:
            self.dict[x] = None

    def update(self, other):
        for x in other:
            self.dict[x] = None

    def union(self, other):
        result = OrderedSet()
        result.update(self)
        result.update(other)
        return result

    def intersection_update(self, other):
        to_be_removed = []
        for x in self:
            if x not in other:
                to_be_removed.append(x)
        for x in to_be_removed:
            self.remove(x)

    def intersection(self, other):
        result = OrderedSet()
        for x in self:
            if x in other:
                result.add(x)
        return result

    def discard(self, element):
        if element in self:
            del self.dict[element]

    def remove(self, element):
        if element not in self:
            raise KeyError(element)
        del self.dict[element]

    def clear(self):
        self.dict.clear()

    def difference(self, other):
        result = OrderedSet()
        for x in self:
            if x not in other:
                result.add(x)
        return result

    def difference_update(self, other):
        for x in other:
            self.discard(x)

    def symmetric_difference(self, other):
        result = OrderedSet()
        for x in self:
            if x not in other:
                result.add(x)
        for x in other:
            if x not in self:
                result.add(x)
        return result

    def __iter__(self):
        for x in self.dict:
            yield x

    def __len__(self):
        return len(self.dict)

    def __contains__(self, element):
        return element in self.dict

    def __repr__(self):
        return "OrderedSet([" + ", ".join(repr(x) for x in self) + "])"

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __sub__(self, other):
        return self.difference(other)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    def __ior__(self, other):
        self.update(other)

    def __iand__(self, other):
        self.intersection_update(other)

    def __isub__(self, other):
        self.difference_update(other)

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return self.dict == other.dict
        return False

    @classmethod
    def __class_getitem__(cls, item):
        return f"{cls.__name__}[{item.__name__}]"


class DisjointDict:
    """A dictionary for recursive lookup.
    Path compression is used to avoid excess of maximum recursion depth."""

    def __init__(self):
        self.values = {}

    def update(self, keys, values):
        for key, value in zip(keys, values):
            self.values[key] = value

    def recursive_lookup(self, key):
        lookup_queue = [key]
        value = None
        while len(lookup_queue) > 0:
            k = lookup_queue.pop()
            if value is not None:
                self.values[k] = value
                continue
            if k not in self.values:
                value = k
                continue
            lookup_queue.append(k)
            lookup_queue.append(self.values[k])
        return value

    def keys(self):
        return list(self.values.keys())


def cached_property(fn, *args, **kwargs):
    """
    Decorator to make a function a "cached property".

    This means that it is a property whose return value is cached after the
    first time it is called.

    Args:
        fn: The function to be made a cached property
        *args: Any args for the function
        **kwargs: Any kwargs for the function
    Returns:
        function
    """
    return property(functools.lru_cache()(fn, *args, **kwargs))


########################################
##### XLA API Utilities
########################################


def get_compile_options(num_replicas: int, num_partitions: int,
                        device_assignment: np.ndarray,
                        use_spmd_partitioning: bool,
                        parameter_is_tupled_arguments: int,
                        build_random_seed: int):
    """Return CompileOptions for XLA compilation."""
    compile_options = xb.get_compile_options(
        num_replicas=num_replicas,
        num_partitions=num_partitions,
        device_assignment=device_assignment,
        use_spmd_partitioning=use_spmd_partitioning,
    )
    compile_options.parameter_is_tupled_arguments = parameter_is_tupled_arguments
    compile_options.executable_build_options.seed = build_random_seed
    return compile_options


def jaxpr_to_hlo_computation(name: str, closed_jaxpr: ClosedJaxpr,
                             donated_invars: Sequence[bool], backend):
    """Convert a jaxpr to a XLA HLO computation.

    Reference code: jax/jax/_src/dispatch.py::lower_xla_callable
    """
    backend_name = backend.platform
    in_avals = [var.aval for var in closed_jaxpr.jaxpr.invars]
    consts = closed_jaxpr.consts
    map(dispatch.prefetch,
        it.chain(consts, dispatch.jaxpr_literals(closed_jaxpr.jaxpr)))

    # Convert jaxpr to XLA HLO
    tuple_args = False
    axis_env = xla.AxisEnv(nreps=1, names=(), sizes=())
    name_stack = xla.new_name_stack(xla.wrap_name(name, 'parallelize'))
    c = xc.XlaBuilder(name)
    xla_consts = xla._xla_consts(c, consts)
    xla_args, donated_invars = xla._xla_callable_args(
        c, in_avals, tuple_args, donated_invars=donated_invars)
    ctx = xla.TranslationContext(c, backend_name, axis_env, name_stack)
    out_nodes = xla.jaxpr_subcomp(ctx, closed_jaxpr.jaxpr, xla_consts,
                                  *xla_args)
    out_tuple = xc.ops.Tuple(c, out_nodes)

    # Set up aliases (donating invars)
    if donated_invars:
        if backend.platform in ("gpu", "tpu"):
            donation_results = xla.set_up_aliases(c, xla_args,
                                                  c.GetShape(out_tuple),
                                                  donated_invars, tuple_args)
        if any(donation_results):
            unused_donations = [
                str(c.GetShape(a))
                for a, d in zip(xla_args, donation_results)
                if d
            ]
            warn_msg = ", ".join(unused_donations)
            warn(f"Some donated buffers were not usable: {warn_msg}")

    return c.build(out_tuple)


def setup_computation_alias(xla_computation: xc.XlaComputation,
                            donated_invars: Sequence[bool]):
    """Set input/output alias in xla computation.

    Assume the tensors in output tuple strictly match the donated parameters.
    """
    program_shape = xla_computation.program_shape()
    parameter_shapes = program_shape.parameter_shapes()
    result_shapes = program_shape.result_shape().tuple_shapes()

    assert len(parameter_shapes) == len(donated_invars), (
        "Zhuohan: This error might be caused by an error in "
        "XLA stage slicing.")

    p_in = 0
    p_out = 0
    while p_in < len(parameter_shapes) and p_out < len(result_shapes):
        if donated_invars[p_in]:
            if parameter_shapes[p_in] == result_shapes[p_out]:
                xla_computation.setup_alias((p_out,), p_in, ())
                p_in += 1
                p_out += 1
            else:
                p_out += 1
        else:
            p_in += 1

    while p_in < len(parameter_shapes):
        if donated_invars[p_in]:
            warn("Some vars are not donated")
        p_in += 1


def count_communication_primitives(hlo_ir: str,
                                   ignore_scalar_all_reduce: bool = False):
    """Count the communication primitives in a HLO IR."""
    total = hlo_ir.count("channel_id")
    all_reduce = hlo_ir.count("all-reduce(") + hlo_ir.count("all-reduce-start(")
    all_gather = hlo_ir.count("all-gather(") + hlo_ir.count("all-gather-start(")
    reduce_scatter = hlo_ir.count("reduce-scatter(") + hlo_ir.count(
        "reduce-scatter-start(")
    all_to_all = hlo_ir.count("all-to-all(") + hlo_ir.count("all-to-all-start(")

    if ignore_scalar_all_reduce:
        # Ignore allreduce of scalar values
        scalar_all_reduce = 0
        scalar_all_reduce += hlo_ir.count("all-reduce(f32[]")
        scalar_all_reduce += hlo_ir.count("all-reduce-start(f32[]")
        scalar_all_reduce += hlo_ir.count("all-reduce(f16[]")
        scalar_all_reduce += hlo_ir.count("all-reduce-start(f16[]")
        total -= scalar_all_reduce
        all_reduce -= scalar_all_reduce

    return total, all_reduce, all_gather, reduce_scatter, all_to_all


def compile_dummy_zero_constant(backend, num_devices: int):
    """Compile an XLA executable that returns a constant zero."""
    c = xc.XlaBuilder("dummy_zero_constant")
    sharding = xc.OpSharding()
    sharding.type = sharding.type.REPLICATED
    c.set_sharding(sharding)
    zero = xc.ops.Constant(c, np.array(0, dtype=np.dtype(np.int32)))
    c.clear_sharding()
    c = c.build(xc.ops.Tuple(c, [zero]))

    compile_options = xb.get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
    )
    compiled = backend.compile(c, compile_options)
    return compiled


def compile_allocate_zero_buffers(backend, num_devices: int,
                                  shapes: Sequence[Sequence[int]],
                                  dtypes: Sequence[jnp.dtype]):
    """Compile an XLA executable that returns zero buffers with given shape and dtypes."""
    c = xc.XlaBuilder("allocate_zero_buffers")
    sharding = xc.OpSharding()
    sharding.type = sharding.type.REPLICATED
    c.set_sharding(sharding)
    ret = []
    for shape, dtype in zip(shapes, dtypes):
        zero = xc.ops.Constant(c, np.array(0, dtype=dtype))
        zero = xc.ops.Broadcast(zero, shape)
        ret.append(zero)
    c.clear_sharding()
    c = c.build(xc.ops.Tuple(c, ret))

    compile_options = xb.get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
    )
    compiled = backend.compile(c, compile_options)
    return compiled


def compile_memset_zero_buffers(backend, num_devices: int,
                                shapes: Sequence[Sequence[int]],
                                dtypes: Sequence[jnp.dtype]):
    """
    Compile an XLA executable that memset zero buffers with given shape and dtypes.
    Try to avoid memcpy
    """
    c = xc.XlaBuilder("allocate_zero_buffers")
    args = []
    sharding = xc.OpSharding()
    sharding.type = sharding.type.REPLICATED
    c.set_sharding(sharding)
    for shape, dtype in zip(shapes, dtypes):
        args.append(
            xc.ops.Parameter(c, len(args),
                             xc.shape_from_pyval(np.ones(shape, dtype))))
    sharding_tuple = xc.OpSharding()
    sharding_tuple.type = sharding.type.TUPLE
    sharding_tuple.tuple_shardings = [sharding for _ in shapes]
    c.set_sharding(sharding_tuple)
    input_params = xc.ops.Tuple(c, args)
    c.set_sharding(sharding)
    output_shape = xc.Shape.scalar_shape(np.dtype(np.float32))
    output_tuple = xc.ops.CustomCall(c,
                                     b'__builtin$MemZero',
                                     operands=(input_params,),
                                     shape=output_shape)
    c = c.build(output_tuple)

    compile_options = xb.get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
    )
    compiled = backend.compile(c, compile_options)
    return compiled


def compile_concatenate(backend, mesh_shape, sharding_spec, batch_size,
                        batch_dim, aval):
    num_devices = np.prod(mesh_shape)
    sharding = pxla.sharding_spec_sharding_proto(sharding_spec)
    build_random_seed = global_config.build_random_seed
    compile_options = get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
        parameter_is_tupled_arguments=False,
        build_random_seed=build_random_seed)
    c = xc.XlaBuilder("concatenate buffers")
    c.set_sharding(sharding)
    operands = []
    for batch_idx in range(batch_size):
        operands.append(
            xc.ops.Parameter(
                c, batch_idx,
                xc.shape_from_pyval(np.ones(aval.shape, aval.dtype))))
    concated = xc.ops.ConcatInDim(c, operands, batch_dim)
    c = c.build(concated)
    compiled = backend.compile(c, compile_options)
    hlo_proto = compiled.hlo_modules()[0].as_serialized_hlo_module_proto()
    return hlo_proto


def get_shard_shape(aval: ShapedArray, sharding_spec: pxla.ShardingSpec):
    """Return the shape of a shard."""
    shape = []
    for dim, spec_dim in zip(aval.shape, sharding_spec.sharding):
        if isinstance(spec_dim, pxla.NoSharding):
            shape.append(dim)
        elif isinstance(spec_dim, pxla.Chunked):
            shape.append(dim // np.prod(spec_dim.chunks))
        elif isinstance(spec_dim, pxla.Unstacked):
            shape.append(spec_dim.size)
    return tuple(shape)


def get_microbatch_sharding_spec(spec: pxla.ShardingSpec, batch_dim,
                                 num_micro_batch):
    batch_dim_chunks = [num_micro_batch]
    if isinstance(spec.sharding[batch_dim], pxla.Chunked):
        batch_dim_chunks.extend(spec.sharding[batch_dim].chunks)
    batch_dim_axis = 0
    for sharding in spec.sharding[:batch_dim]:
        if isinstance(sharding, pxla.Chunked):
            batch_dim_axis += 1

    new_sharding = list(spec.sharding)
    new_sharding[batch_dim] = pxla.Chunked(batch_dim_chunks)

    new_mapping = []
    for mapping in spec.mesh_mapping:
        if isinstance(mapping, pxla.Replicated):
            new_mapping.append(mapping)
            continue
        assert isinstance(mapping, pxla.ShardedAxis)
        new_axis = mapping.axis
        if mapping.axis >= batch_dim_axis:
            new_axis += 1
        new_mapping.append(pxla.ShardedAxis(new_axis))
    new_mapping.append(pxla.ShardedAxis(batch_dim_axis))

    return pxla.ShardingSpec(sharding=tuple(new_sharding),
                             mesh_mapping=tuple(new_mapping))


class XlaPassContext:
    """A global context for passing arguments from python to XLA c++ passes."""

    current = None

    def __init__(self, value_dict):
        self.value_dict = value_dict

    def __enter__(self):
        assert XlaPassContext.current is None, "Do not support recurrent context"
        XlaPassContext.current = self
        xe.set_pass_context(self.value_dict)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        XlaPassContext.current = None
        xe.clear_pass_context()


########################################
##### Jaxpr Utilities
########################################
def clone_jaxpr(closed_jaxpr: ClosedJaxpr,
                invars: Sequence[Atom] = None,
                outvars: Sequence[Var] = None,
                eqns: Sequence[JaxprEqn] = None,
                constvars: Sequence[Var] = None,
                consts: Sequence = None):
    """Clone a jaxpr and replace members if they are provided."""
    constvars = constvars or closed_jaxpr.jaxpr.constvars
    invars = invars or closed_jaxpr.jaxpr.invars
    outvars = outvars or closed_jaxpr.jaxpr.outvars
    eqns = eqns or closed_jaxpr.jaxpr.eqns
    consts = consts or closed_jaxpr.consts
    jaxpr = Jaxpr(constvars, invars, outvars, eqns)
    return ClosedJaxpr(jaxpr, consts)


def trace_jaxpr_with_micro_batch(fun, batch_invars, num_micro_batches,
                                 raw_avals):
    """Trace the jaxpr of the computation of a micro batch."""
    avals = []
    batch_size = None
    for aval, is_batch_var in zip(raw_avals, batch_invars):
        if is_batch_var:
            assert aval.shape[0] % num_micro_batches == 0, (
                "The batch dimension must be divisable by num_micro_batches.")
            if batch_size is None:
                batch_size = aval.shape[0] // num_micro_batches
            else:
                assert batch_size == aval.shape[0] // num_micro_batches, (
                    "The batch dimension must be the same for all batch vars.")
            shape = (batch_size,) + aval.shape[1:]
            avals.append(aval.update(shape=shape))
        else:
            avals.append(aval)
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    return closed_jaxpr, avals, batch_size


def slices_to_jaxpr(closed_jaxpr: ClosedJaxpr,
                    sliced_eqns) -> Sequence[ClosedJaxpr]:
    """Wrap sliced equations to a list of ClosedJaxpr."""
    n_eqns = len(sliced_eqns)
    global_invars = OrderedSet(closed_jaxpr.jaxpr.invars)
    global_consts = dict(zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))
    global_outvars = OrderedSet(
        var for var in closed_jaxpr.jaxpr.outvars if isinstance(var, Var))
    result = []
    layer_invars = [OrderedSet() for _ in range(n_eqns)]
    layer_outvars = [OrderedSet() for _ in range(n_eqns)]
    layer_consts = [{} for _ in range(n_eqns)]
    var_layer_dict = {}
    for i, eqns in enumerate(sliced_eqns):
        for eqn in eqns:
            for var in eqn.invars:
                if isinstance(var, Literal):
                    continue
                if var in global_consts:
                    layer_consts[i][var] = global_consts[var]
                elif var in global_invars:
                    layer_invars[i].add(var)
                elif var_layer_dict[var] != i:
                    layer_invars[i].add(var)
                    layer_outvars[var_layer_dict[var]].add(var)
                else:
                    assert var_layer_dict[var] == i
            for var in eqn.outvars:
                if not isinstance(var, DropVar):
                    var_layer_dict[var] = i
                if var in global_outvars:
                    layer_outvars[i].add(var)
    for i, eqns in enumerate(sliced_eqns):
        new_jaxpr = Jaxpr(list(layer_consts[i].keys()), list(layer_invars[i]),
                          list(layer_outvars[i]), eqns)
        new_closed_jaxpr = ClosedJaxpr(new_jaxpr,
                                       list(layer_consts[i].values()))
        result.append(new_closed_jaxpr)
    return result


def log_jaxpr(jaxpr: ClosedJaxpr, filename: str):
    """Print jaxpr int a temporary file for debugging purposes."""
    path = "/tmp/" + filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(jaxpr))


########################################
##### Profiling Utilities
########################################


def profile_xla_executable(compiled, backend, local_devices):
    """Measure the time costs of a xla executable with dummy inputs."""
    hlo_module = compiled.hlo_modules()[0]
    cost_failed = [np.inf] * 3

    # Allocate dummy buffers
    input_shapes = hlo_module.parameter_shapes()

    # prune OOM cases, not exact because third party lib not considered:
    free_mem = local_devices[0].available_memory()
    input_bytes = 0
    for shape in input_shapes:
        input_bytes += np.prod(
            shape.dimensions()) * shape.numpy_dtype().itemsize
    if free_mem < compiled.total_allocation_size() and free_mem != -1:
        return cost_failed

    device_inputs = []
    try:
        for shape in input_shapes:
            device_inputs.append([
                backend.buffer_from_pyval(
                    np.empty(shape.dimensions(), shape.numpy_dtype()), device)
                for device in local_devices
            ])
        local_devices[0].synchronize_all_activity()
    except RuntimeError:
        return cost_failed

    # Run benchmark
    def run_func():
        device_outputs = compiled.execute_sharded_on_local_devices(
            device_inputs)

        # Reset the value for donate buffers
        ct = 0
        for j in range(len(device_inputs)):
            if device_inputs[j][0].is_deleted():
                device_inputs[j] = device_outputs[ct]
                ct += 1

        local_devices[0].synchronize_all_activity()

    try:
        costs = benchmark_func(run_func, repeat=3, number=3)
    except RuntimeError:
        costs = cost_failed
    return costs


def benchmark_func(run_func,
                   sync_func=None,
                   warmup=1,
                   repeat=3,
                   number=5,
                   min_repeat_second=None):
    """
    Benchmark the execution time of a function.

    The function is executed for (warmup + number * repeat) times.
    The return value is a list of `repeat` elements and each elements is
    the average execution time of `number` executions.

    If `min_repeat_second` is set, the function automatically picks a `number`
    so that one `repeat` lasts for at least `min_repeat_second` seconds.
    """
    costs = []

    # Warmup
    for _ in range(warmup):
        run_func()

    # Choose a "number" according to "min_repeat_second"
    if min_repeat_second:
        if sync_func:
            sync_func()
        tic = time.time()
        run_func()
        if sync_func:
            sync_func()
        toc = time.time()
        cost = toc - tic
        number = max(int(min_repeat_second / cost), 1)

    # Benchmark
    for _ in range(repeat):
        if sync_func:
            sync_func()
        tic = time.time()
        for __ in range(number):
            run_func()
        if sync_func:
            sync_func()
        costs.append(time.time() - tic)

    return np.array(costs) / number


########################################
##### Array conversion
########################################


def is_continuous_subset(tensor_slice, tensor_shape, row_major=True):
    """
    Figure out whether a slice is a continuous subset of the tensor.

    Args:
        slice_shape (Sequence(slice)): the shape of the slice.
        tensor_shape (Sequence(int)): the shape of the tensor.
        row_major (bool): whether the tensor layout is row-majored.

    Returns:
        is_continuous (bool)
    """
    if not row_major:
        raise NotImplementedError("Do not support column major.")
    ndim = len(tensor_shape)
    if len(tensor_slice) != ndim:
        raise RuntimeError("ndims mismatch.")
    slice_shape = tuple(ind.stop - ind.start for ind in tensor_slice)
    for dim, dim_shape in enumerate(slice_shape):
        if dim + 1 > ndim:
            return True
        if dim_shape == 1:
            continue
        return slice_shape[dim + 1:] == tensor_shape[dim + 1:]


def infer_offset_and_n_elements(tensor_slice):
    """Calculate the offset and #elements before making NCCL calls.

    This function assumes the slice is a continuous subset of the original tensor.
    """
    slice_shape = tuple(ind.stop - ind.start for ind in tensor_slice)
    offset = tuple()
    n_elements = np.prod(slice_shape)
    for dim, dim_shape in enumerate(slice_shape):
        offset = offset + (tensor_slice[dim].start,)
        if dim_shape > 1:
            break
    return offset, n_elements


def xla_buffer_to_jax_tensor(xla_buf):
    """
    Convert an xla buffer to a JAX DeviceArray.

    So we can index over the data buffer.
    """
    aval = ShapedArray(xla_buf.shape, xla_buf.dtype)
    return _DeviceArray(aval, xla_buf.device(), xla_buf)


def jax_tensor_to_xla_buffer(jax_buf):
    """Convert a JAX Device array back to XLA buffer."""
    return jax_buf.device_buffer


def xla_buffer_to_cupy(xla_buf, take_ownership=False):
    """Convert an xla buffer directly to cupy, w/o transitioning from jax buffer."""
    return cp.fromDlpack(
        xc._xla.buffer_to_dlpack_managed_tensor(xla_buf,
                                                take_ownership=take_ownership))


def cupy_to_xla_buffer(tensor):
    """Convert cupy tensors to XLA buffers."""
    if isinstance(tensor, list):
        return list(map(cupy_to_xla_buffer, tensor))
    cpu_backend = xb.get_backend("cpu")
    try:
        gpu_backend = xb.get_backend("gpu")
    except RuntimeError:
        gpu_backend = None
    buf = xc._xla.dlpack_managed_tensor_to_buffer(tensor.toDlpack(),
                                                  cpu_backend, gpu_backend)
    return buf


def jax_tensor_to_cupy(tensors, take_ownership=False):
    """Convert a Jax DeviceArray to cupy tensor; zero copy."""
    if isinstance(tensors, list):
        return list(map(jax_tensor_to_cupy, tensors))
    return cp.fromDlpack(to_dlpack(tensors, take_ownership=take_ownership))


def cupy_to_jax_tensor(tensors):
    """Convert cupy tensors to JAX tensors."""
    if isinstance(tensors, list):
        return list(map(cupy_to_jax_tensor, tensors))
    return from_dlpack(tensors.toDlpack())


# Note: use Python jit instead of CPP jit,
# because CPP jit has bugs on _DeviceArray.
if is_worker:
    FLAGS.experimental_cpp_jit = False


# Note(Hao): this function will be jit-ed into as many versions as the possible length of start_indices
@partial(jax.jit, donate_argnums=0, static_argnums=2)
def jax_tensor_set(src_buf, update, start_indices):
    """
    In-place write on a JAX buffer.

    Args:
        src_buf: JAX device array.
        update: JAX device array.
        start_indices (tuple[int]): tuple of integers indicating the starting indices.
    """
    # src_buf = src_buf.at[indices].set(update)
    src_buf = jax.lax.dynamic_update_slice(src_buf, update, start_indices)
    return src_buf


@partial(jax.jit, static_argnums=(1, 2))
def jax_tensor_index(src_tensor, indices, size):
    dst_tensor = jax.lax.dynamic_slice(src_tensor, indices, size)
    return dst_tensor


########################################
##### OS / IO Utilities
########################################


def run_cmd(cmd: str):
    """Run a bash command."""
    print(cmd)
    ret = os.system(cmd)
    return ret


def list_gpu_info():
    """List all gpu information by calling nvidia-sim."""
    ret = subprocess.getoutput("nvidia-smi -L")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible_devices:
        ids = [int(x) for x in visible_devices.split(",")]
        lines = ret.split("\n")
        lines = [lines[i] for i in ids]
        ret = "\n".join(lines)
    return ret


def disable_tqdm_globally():
    """Disable tqdm globally."""
    tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)


def get_num_hosts_and_num_devices(args):
    """Get the number of hosts and the number of devices per host for benchmark scripts."""
    if args.num_hosts is not None or args.num_devices_per_host is not None:
        assert args.num_hosts is not None and args.num_devices_per_host is not None
        num_hosts, num_devices_per_host = args.num_hosts, args.num_devices_per_host
    else:
        if hasattr(args, "local") and args.local:
            num_hosts = 1
            num_devices_per_host = list_gpu_info().count("UUID")
        else:
            ray.init(address="auto", namespace=get_ray_namespace_str())
            num_hosts = len(ray.nodes())
            num_devices_per_host = int(
                ray.cluster_resources()["GPU"]) // num_hosts
    return num_hosts, num_devices_per_host


def get_ray_namespace_str(prefix=global_config.default_ray_namespace_prefix):
    """Get a unique ray namespace str to avoid some annoyed warnings."""
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    namespace_str = f"{prefix}-{date_str}"
    return namespace_str


def write_tsv(heads: Sequence[str],
              values: Sequence[Any],
              filename: str,
              print_line: bool = True):
    """Write tsv data to a file."""
    assert len(heads) == len(values)

    values = [str(x) for x in values]

    with open(filename, "a", encoding="utf-8") as fout:
        fout.write("\t".join(values) + "\n")

    if print_line:
        line = ""
        for i in range(len(heads)):
            line += heads[i] + ": " + values[i] + "  "
        print(line)


def to_str_round(x: Any, decimal: int = 6):
    """Print a python object but round all floating point numbers."""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        tmp_str = ", ".join([to_str_round(y, decimal=decimal) for y in x])
        return "[" + tmp_str + "]"
    if isinstance(x, dict):
        return str({k: to_str_round(v, decimal=decimal) for k, v in x.items()})
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        format_str = f"%.{decimal}f"
        return format_str % x
    if x is None:
        return str(x)
    raise ValueError("Invalid value: " + str(x))


_tic = None


def print_used_time(message: str):
    """Print a message and the elapsed time from the last call."""
    global _tic
    if message:
        print(f" - {message}: {time.time() - _tic:.2f} s")
    _tic = time.time()


########################################
##### Other Utilities
########################################

GB = 1 << 30  # Gigabyte
MB = 1 << 20  # Megabyte


def map_to_shape(array_pytree: PyTreeDef):
    """Map a PyTree of jax arrays to their shapes."""
    return tree_map(lambda x: getattr(x, "shape", None), array_pytree)


def compute_bytes(pytree: PyTreeDef):
    """Compute the total bytes of arrays in a pytree."""
    flatten_args, _ = tree_flatten(pytree)
    ret = 0
    for x in flatten_args:
        if hasattr(x, "shape"):
            ret += np.prod(x.shape) * x.dtype.itemsize
    return ret


def compute_param_number(pytree: PyTreeDef):
    """Compute the total number of elements in a pytree."""
    flatten_args, _ = tree_flatten(pytree)
    ret = 0
    for x in flatten_args:
        if hasattr(x, "shape"):
            ret += np.prod(x.shape)
    return ret


def get_var_mapping(mapping, var):
    """map the var to a new value if var is Var and in the mapping."""
    if isinstance(var, Var) and var in mapping:
        return mapping[var]
    else:
        return var

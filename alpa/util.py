# pylint: disable=consider-using-enumerate
"""Common utilities."""
import functools
import itertools as it
import logging
import os
import subprocess
import re
import time
from collections import OrderedDict
from functools import partial, partialmethod
import threading
from typing import Iterable, Dict, Sequence, Any, List
from warnings import warn

from flax.training import train_state
from flax.training.common_utils import stack_forest
import jax
from jax._src.source_info_util import SourceInfo
import jax.numpy as jnp
from jax._src import dispatch, util
from jax._src.api import FLAGS, ShapeDtypeStruct
from jax._src.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
from jax.api_util import shaped_abstractify
from jax import core
from jax.core import (Atom, ClosedJaxpr, DropVar, Jaxpr, JaxprEqn, Literal,
                      Primitive, ShapedArray, Var, AbstractValue, gensym)
from jax.experimental.maps import FrozenDict
from jax import linear_util as lu
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla, pxla, mlir
from jax.interpreters.xla import _DeviceArray
from jax.tree_util import tree_map, tree_flatten, PyTreeDef
import numpy as np
import ray
from ray.util.placement_group import get_current_placement_group,\
    PlacementGroup
import tqdm

from alpa import device_mesh
from alpa.global_env import global_config, is_worker
from alpa.monkey_patch import (restore_random, monkey_patch_random,
                               rng_primitives)
from alpa.wrapped_hlo import HloStatus, WrappedHlo

PLACEMENT_GROUP_TIMEOUT_S_ENV = "ALPA_PLACEMENT_GROUP_TIMEOUT_S_ENV"

########################################
##### Alpa API Utilities
########################################

logger = logging.getLogger(__name__)


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

        if isinstance(arg, train_state.TrainState):
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
        if isinstance(x, train_state.TrainState):
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


def update_jax_platform(platform):
    """Update the jax backend platform."""
    jax.config.update("jax_platform_name", platform)
    xb.get_backend.cache_clear()


class GradFuncTransformContext:
    """
    A context to hold transformations applied to the forward function
    before calling alpa.grad or alpa.value_and_grad.
    """
    transforms = []

    def __init__(self, transform):
        self.transform = transform

    def __enter__(self):
        GradFuncTransformContext.transforms.append(self.transform)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        GradFuncTransformContext.transforms.pop()


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
        self.dict.update({x: None for x in iterable})

    def add(self, *args):
        self.dict.update({x: None for x in args})

    def update(self, other):
        self.dict.update({x: None for x in other})

    def union(self, other):
        result = OrderedSet(self)
        result.update(other)
        return result

    def intersection_update(self, other):
        for x in [x for x in self.dict if x not in other]:
            del self.dict[x]

    def intersection(self, other):
        return OrderedSet(x for x in self if x in other)

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
        return OrderedSet([x for x in self if x not in other])

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
        return iter(self.dict)

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
        if not isinstance(keys, Iterable):
            assert not isinstance(values, Iterable)
            self.values[keys] = values
            return
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


def get_compile_options(num_replicas: int,
                        num_partitions: int,
                        device_assignment: np.ndarray,
                        use_spmd_partitioning: bool,
                        parameter_is_tupled_arguments: int,
                        build_random_seed: int,
                        spmd_propagation_to_outputs: bool = False):
    """Return CompileOptions for XLA compilation."""
    compile_options = xb.get_compile_options(
        num_replicas=num_replicas,
        num_partitions=num_partitions,
        device_assignment=device_assignment,
        use_spmd_partitioning=use_spmd_partitioning,
    )
    compile_options.parameter_is_tupled_arguments = (
        parameter_is_tupled_arguments)
    build_options = compile_options.executable_build_options
    build_options.seed = build_random_seed
    build_options.allow_spmd_sharding_propagation_to_output =\
        spmd_propagation_to_outputs
    return compile_options


def jaxpr_to_hlo(name: str,
                 closed_jaxpr: ClosedJaxpr,
                 donated_invars: Sequence[bool],
                 platform: str = "cuda"):
    """Convert a jaxpr to a wrapped XLA HloModule.

    Reference code: jax/jax/_src/dispatch.py::lower_xla_callable
    """
    consts = closed_jaxpr.consts
    map(dispatch.prefetch,
        it.chain(consts, dispatch.jaxpr_literals(closed_jaxpr.jaxpr)))

    # Convert jaxpr to XLA HLO
    tuple_args = False
    axis_env = xla.AxisEnv(nreps=1, names=(), sizes=())
    name_stack = util.new_name_stack(xla.wrap_name(name, "parallelize"))
    closed_jaxpr = ClosedJaxpr(closed_jaxpr.jaxpr, consts)
    unordered_effects = [
        eff for eff in closed_jaxpr.effects if eff not in core.ordered_effects
    ]
    ordered_effects = [
        eff for eff in closed_jaxpr.effects if eff in core.ordered_effects
    ]
    lowering_result = mlir.lower_jaxpr_to_module(
        name, closed_jaxpr, unordered_effects, ordered_effects, None, platform,
        mlir.ReplicaAxisContext(axis_env), name_stack, donated_invars)
    xla_computation = xe.mlir.mlir_module_to_xla_computation(
        mlir.module_to_string(lowering_result.module),
        use_tuple_args=tuple_args,
        return_tuple=True)
    return WrappedHlo(xla_computation)


def setup_computation_alias(hlo: WrappedHlo, donated_invars: Sequence[bool]):
    """Set input/output alias in xla computation.

    Assume the tensors in output tuple strictly match the donated parameters.
    """
    program_shape = hlo.program_shape()
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
                hlo.get_module().setup_alias((p_out,), p_in, ())
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


def compile_dummy_zero_constant():
    """Compile an Hlo module that returns a constant zero."""
    c = xc.XlaBuilder("dummy_zero_constant")
    sharding = xc.OpSharding()
    sharding.type = sharding.type.REPLICATED
    c.set_sharding(sharding)
    zero = xc.ops.Constant(c, np.array(0, dtype=np.dtype(np.int32)))
    c.clear_sharding()
    c = c.build(xc.ops.Tuple(c, [zero]))
    return WrappedHlo(c, HloStatus.SHARDING_ANNOTATED)


def compile_allocate_zero_buffers(backend, num_devices: int,
                                  shapes: Sequence[Sequence[int]],
                                  dtypes: Sequence[jnp.dtype]):
    """Compile an XLA executable that returns zero buffers with given shape and
    dtypes."""
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
    with XlaPassContext({
            "done-event::enable": global_config.enable_overlapping,
    }):
        compiled = backend.compile(c, compile_options)
    return compiled


def compile_concatenate(mesh_shape, sharding_spec, batch_size, batch_dim, aval):
    """
    Compile an XLA executable that concatenates values over the batch dimension,
    keeping the sharding spec unchanged.
    """
    c = xc.XlaBuilder("concatenate buffers")
    sharding = pxla.sharding_spec_sharding_proto(sharding_spec)
    c.set_sharding(sharding)
    operands = []
    for batch_idx in range(batch_size):
        operands.append(
            xc.ops.Parameter(
                c, batch_idx,
                xc.shape_from_pyval(np.ones(aval.shape, aval.dtype))))
    concated = xc.ops.ConcatInDim(c, operands, batch_dim)
    hlo_module = c.build(concated).as_hlo_module()

    num_devices = np.prod(mesh_shape)
    build_random_seed = global_config.compile_random_seed
    compile_options = get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
        parameter_is_tupled_arguments=False,
        build_random_seed=build_random_seed)
    xe.run_spmd_partitioner(hlo_module, compile_options)
    return WrappedHlo(hlo_module, HloStatus.SPMD_PARTITIONED)


def compile_allgather(shape, dtype, src_spec, dst_spec, num_devices):
    """
    Compile an XLA executable that runs allgather to reshard the tensor from src
    sharding spec to dst sharding spec.
    """
    c = xc.XlaBuilder("allgather")
    src_sharding = pxla.sharding_spec_sharding_proto(src_spec)
    c.set_sharding(src_sharding)
    operand = xc.ops.Parameter(c, 0, xc.shape_from_pyval(np.ones(shape, dtype)))
    c.clear_sharding()

    dst_sharding = xc.OpSharding()
    dst_sharding.type = dst_sharding.type.TUPLE
    dst_sharding.tuple_shardings = [pxla.sharding_spec_sharding_proto(dst_spec)]

    c.set_sharding(dst_sharding)
    hlo_module = c.build(xc.ops.Tuple(c, [operand])).as_hlo_module()

    build_random_seed = global_config.compile_random_seed
    compile_options = get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
        parameter_is_tupled_arguments=False,
        build_random_seed=build_random_seed)
    xe.run_spmd_partitioner(hlo_module, compile_options)
    return WrappedHlo(hlo_module, HloStatus.SPMD_PARTITIONED)


def get_index_select_computation(sharding_specs, dim, avals, index_shape):
    """Compile an XLA executable that runs index select for each tensor."""
    c = xc.XlaBuilder("index_select")
    shardings = []
    selected = []
    index = xc.ops.Parameter(c, len(avals), index_shape)
    for i, aval in enumerate(avals):
        sharding_spec = sharding_specs[i]
        sharding = pxla.sharding_spec_sharding_proto(sharding_spec)
        c.set_sharding(sharding)
        operand = xc.ops.Parameter(
            c, i, xc.shape_from_pyval(np.ones(aval.shape, aval.dtype)))
        c.clear_sharding()
        index_selected = xc.ops.IndexSelect(operand, index, dim)
        shardings.append(sharding)
        selected.append(index_selected)
    sharding2 = xc.OpSharding()
    sharding2.type = sharding.type.TUPLE
    sharding2.tuple_shardings = shardings
    c.set_sharding(sharding2)
    c = c.build(xc.ops.Tuple(c, selected))
    return WrappedHlo(c, HloStatus.SHARDING_ANNOTATED)


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
        assert XlaPassContext.current is None, ("Do not support nested context")
        XlaPassContext.current = self
        xe.set_pass_context(self.value_dict)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        XlaPassContext.current = None
        xe.clear_pass_context()


def undefined_sharding_spec_proto():
    """Return a proto of ShardingSpec which represents an undefined spec."""
    # We reuse "Manual" to represent "Undefined"
    proto = xc.OpSharding()
    proto.type = xc.OpSharding.Type.MANUAL
    return proto


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
    constvars = closed_jaxpr.jaxpr.constvars if constvars is None else constvars
    invars = closed_jaxpr.jaxpr.invars if invars is None else invars
    outvars = closed_jaxpr.jaxpr.outvars if outvars is None else outvars
    eqns = closed_jaxpr.jaxpr.eqns if eqns is None else eqns
    consts = closed_jaxpr.consts if consts is None else consts
    jaxpr = Jaxpr(constvars, invars, outvars, eqns)
    return ClosedJaxpr(jaxpr, consts)


def new_jaxpr_eqn(invars,
                  outvars,
                  primitive,
                  params,
                  effects=None,
                  source_info=None):
    """Create a new jaxpr equation."""
    effects = effects or core.no_effects
    return core.new_jaxpr_eqn(invars, outvars, primitive, params, effects,
                              source_info)


def clone_jaxpr_eqn(eqn: JaxprEqn,
                    invars: Sequence[Atom] = None,
                    outvars: Sequence[Var] = None,
                    primitive: Primitive = None,
                    params: Dict[str, Any] = None,
                    effects: Any = None,
                    source_info: SourceInfo = None):
    invars = list(invars or eqn.invars)
    outvars = list(outvars or eqn.outvars)
    primitive = primitive or eqn.primitive
    params = dict(params or eqn.params)
    source_info = source_info or eqn.source_info
    effects = effects or eqn.effects
    return new_jaxpr_eqn(invars, outvars, primitive, params, effects,
                         source_info)


def process_remat(closed_jaxpr: ClosedJaxpr):
    """Offload remat call from forward to backward.

    remat in Jax generates some remat_call in the forward part, but these
    remat_call only outputs constant and does not rely on inputs.
    Hence, offloading them into the backward part does not enlong any liveness
    interval, while helps reduce forward output size.

    As Alpa monkey patches random number generation to stateful version,
    this function also gets the generated rng state and set it an input
    of the offloaded remat part.

    Args:
        closed_jaxpr: the original jaxpr.

    Returns:
        new_jaxpr: the processed jaxpr
    """
    # pylint: disable=import-outside-toplevel
    from alpa.pipeline_parallel.primitive_def import pipeline_p

    def only_create_consts(jaxpr: Jaxpr):
        const_vars = OrderedSet()
        for eqn in jaxpr.eqns:
            for var in eqn.invars:
                if isinstance(var, Var) and var not in const_vars:
                    return False
            const_vars.update(
                [v for v in eqn.outvars if not isinstance(v, DropVar)])
        return True

    def only_input_consts(eqn: JaxprEqn):
        in_bytes = 0
        for var in eqn.invars:
            if not isinstance(var, Var):
                continue
            if isinstance(var, DropVar):
                continue
            in_bytes += np.prod(var.aval.shape) * np.dtype(
                var.aval.dtype).itemsize
        return in_bytes == 0

    def is_meaningful(inv: Atom):
        return isinstance(inv, Var) and not isinstance(inv, DropVar)

    def _offload_remat_process_pipeline(eqn: JaxprEqn,
                                        discard_invars: Sequence[Var]):
        discard_invars = set(discard_invars)
        new_invars = []
        new_outvars = []
        for inv, outv in zip(eqn.invars, eqn.outvars):
            if not (is_meaningful(inv) and inv in discard_invars):
                new_invars.append(inv)
                new_outvars.append(outv)
        return clone_jaxpr_eqn(eqn, new_invars, new_outvars)

    def difference_cross_marker(eqns, base, dif):
        base = set(base)
        dif = set(v for v in dif if is_meaningful(v))
        pipeline_mapping = {}
        for eqn in eqns:
            if eqn.primitive is pipeline_p:
                for inv, outv in zip(eqn.invars, eqn.outvars):
                    if is_meaningful(inv) and is_meaningful(outv):
                        pipeline_mapping[outv] = inv
        for var in dif:
            base.discard(var)
            while var in pipeline_mapping:
                var = pipeline_mapping[var]
                base.discard(var)
        return base

    rng_primitives_set = set(rng_primitives)

    def add_rng_as_output(jaxpr: Jaxpr):
        rng_outvars = []
        for eqn in jaxpr.eqns:
            if eqn.primitive in rng_primitives_set:
                assert not eqn.primitive.multiple_results
                rng_outvars.append(eqn.outvars[0])
        new_outvars = jaxpr.outvars + rng_outvars
        return Jaxpr(jaxpr.constvars, jaxpr.invars, new_outvars,
                     jaxpr.eqns), rng_outvars

    def get_rng_from_input(jaxpr: Jaxpr):
        new_invars = list(jaxpr.invars)
        new_eqns = []
        for eqn in jaxpr.eqns:
            if eqn.primitive in rng_primitives_set:
                new_invars.append(eqn.outvars[0])
            else:
                new_eqns.append(eqn)
        return Jaxpr(jaxpr.constvars, new_invars, jaxpr.outvars, new_eqns)

    def clone_outvars(outvars):
        new_outvars = []
        var_mapping = {}
        for v in outvars:
            if isinstance(v, DropVar):
                new_outvars.append(v)
            else:
                new_v = gensym_fn(v.aval)
                new_outvars.append(new_v)
                var_mapping[v] = new_v
                while v in var_pipeline_mapping:
                    v = var_pipeline_mapping[v]
                    var_mapping[v] = new_v
        return new_outvars, var_mapping

    # Find offloaded eqns
    offloaded_eqns = set()
    gensym_fn = gensym([closed_jaxpr.jaxpr])

    for eqn_idx, eqn in enumerate(closed_jaxpr.eqns):
        if (eqn.primitive == pe.remat_call_p and only_input_consts(eqn) and
                only_create_consts(eqn.params["call_jaxpr"])):
            offloaded_eqns.add(eqn_idx)
    # Find where each eqn is offloaded
    # A faster way is to rewrite remat to set each call's name unique, but users
    # may use 'from jax import remat' instead of 'jax.remat()' which disables
    # monkey patch to remat.
    # Dict[fwd_outvar -> fwd_remat_call_idx]
    offloaded_vars_from = {}
    # Dict[var -> var]
    var_pipeline_mapping = {}
    # Dict[bwd_remat_call_idx -> fwd_remat_call_idx]
    offload_to = {}
    for eqn_idx in offloaded_eqns:
        for var in closed_jaxpr.eqns[eqn_idx].outvars:
            if is_meaningful(var):
                offloaded_vars_from[var] = eqn_idx
    for eqn_idx, eqn in enumerate(closed_jaxpr.eqns):
        if (eqn.primitive == pe.remat_call_p and eqn.params["differentiated"]):
            for inv in eqn.invars:
                if is_meaningful(inv) and inv in offloaded_vars_from:
                    fwd_eqn_idx = offloaded_vars_from[inv]
                    assert (eqn_idx not in offload_to or
                            offload_to[eqn_idx] == fwd_eqn_idx
                           ), "A backward matches multiple forward."
                    offload_to[eqn_idx] = fwd_eqn_idx
        elif eqn.primitive == pipeline_p:
            for inv, outv in zip(eqn.invars, eqn.outvars):
                if is_meaningful(inv) and inv in offloaded_vars_from:
                    offloaded_vars_from[outv] = eqn
                    var_pipeline_mapping[inv] = outv
    # Insert the fwd remat call and rewrite corresponding bwd remat call
    new_eqns = []
    discarded = difference_cross_marker(closed_jaxpr.eqns,
                                        offloaded_vars_from.keys(),
                                        closed_jaxpr.jaxpr.outvars)
    # Dict[fwd_eqn_idx -> Sequence[fwd_rng_outvars]]
    rng_vars = {}
    for eqn_idx, eqn in enumerate(closed_jaxpr.eqns):
        if eqn.primitive is pipeline_p:
            # Rewrite pipeline_markers
            new_eqns.append(_offload_remat_process_pipeline(eqn, discarded))
        elif eqn_idx in offloaded_eqns:
            # add rng result as an output
            new_params = dict(eqn.params)
            new_called, rng_outvars = add_rng_as_output(
                new_params["call_jaxpr"])
            new_params["call_jaxpr"] = new_called
            rng_outvars = [gensym_fn(v.aval) for v in rng_outvars]
            new_outvars = list(eqn.outvars) + rng_outvars
            rng_vars[eqn_idx] = rng_outvars
            cloned_eqn = clone_jaxpr_eqn(eqn,
                                         outvars=new_outvars,
                                         params=new_params)
            new_eqns.append(cloned_eqn)
        elif eqn_idx not in offload_to:
            new_eqns.append(eqn)
        else:
            inserted_idx = offload_to[eqn_idx]
            # clone the forward remat call
            # rewrite the inserted. Remove its rng, add invars from the cloned
            inserted = closed_jaxpr.eqns[inserted_idx]
            cloned_invars = list(inserted.invars)
            cloned_invars.extend(rng_vars[inserted_idx])
            cloned_params = dict(inserted.params)
            cloned_params["call_jaxpr"] = get_rng_from_input(
                inserted.params["call_jaxpr"])
            cloned_outvars, var_mapping = clone_outvars(inserted.outvars)
            cloned_fwd = clone_jaxpr_eqn(inserted,
                                         cloned_invars,
                                         cloned_outvars,
                                         params=cloned_params)
            # rewrite invars for bwd remat call
            new_invars = [get_var_mapping(var_mapping, v) for v in eqn.invars]
            new_eqn = clone_jaxpr_eqn(eqn, invars=new_invars)
            new_eqns.extend([cloned_fwd, new_eqn])
    return clone_jaxpr(closed_jaxpr, eqns=new_eqns)


def trace_jaxpr_with_micro_batch(fun: lu.WrappedFun,
                                 batch_invars: Sequence[bool],
                                 num_micro_batches: int,
                                 raw_avals: Sequence[AbstractValue],
                                 batch_dim: int = 0):
    """Trace the jaxpr of the computation of a micro batch."""
    assert batch_dim == 0, "Only support batch_dim == 0"
    # Monkey patch jax.random to fast stateful version
    monkey_patch_random()
    monkey_patch_jaxarray()

    avals = []
    batch_size = None
    for aval, is_batch_var in zip(raw_avals, batch_invars):
        if is_batch_var:
            assert aval.shape[0] % num_micro_batches == 0, (
                f"The batch size must be divisable by num_micro_batches. "
                f"batch_size = {aval.shape[0]}, "
                f"num_micro_batches = {num_micro_batches}")
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

    # Restore jax.random to original stateless version
    restore_random()
    restore_jaxarray()
    return closed_jaxpr, batch_size


backup_jnp_array = jnp.array


def monkey_patch_jaxarray():
    """Monkey patch jnp.array as jnp.asarray to avoid unnecessary copy."""
    jnp.array = jnp.asarray
    setattr(Literal, "__hash__", lambda self: self.hash)


def restore_jaxarray():
    """Monkey patch jnp.array as jnp.asarray to avoid unnecessary copy."""
    jnp.array = backup_jnp_array
    setattr(Literal, "__hash__", None)


def slices_to_jaxpr(
        closed_jaxpr: ClosedJaxpr,
        sliced_eqns: Sequence[Sequence[JaxprEqn]]) -> Sequence[ClosedJaxpr]:
    """Wrap sliced equations to a list of ClosedJaxpr."""
    n_eqns = len(sliced_eqns)
    global_invars = OrderedSet(closed_jaxpr.jaxpr.invars)
    global_outvars = OrderedSet(
        var for var in closed_jaxpr.jaxpr.outvars if isinstance(var, Var))
    global_consts = dict(zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))

    layer_invars = [OrderedSet() for _ in range(n_eqns)]
    layer_outvars = [OrderedSet() for _ in range(n_eqns)]
    layer_consts = [{} for _ in range(n_eqns)]

    var_layer_dict = {}  # Dict[var -> layer_idx]
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

    result = []
    for i, eqns in enumerate(sliced_eqns):
        new_jaxpr = Jaxpr(list(layer_consts[i].keys()), list(layer_invars[i]),
                          list(layer_outvars[i]), eqns)
        new_closed_jaxpr = ClosedJaxpr(new_jaxpr,
                                       list(layer_consts[i].values()))
        result.append(new_closed_jaxpr)
    return result


def get_var_mapping(mapping, var):
    """map the var to a new value if var is Var and in the mapping."""
    if isinstance(var, Var) and var in mapping:
        return mapping[var]
    else:
        return var


def log_jaxpr(jaxpr: ClosedJaxpr, filename: str):
    """Print jaxpr int a temporary file for debugging purposes."""
    path = "/tmp/" + filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(jaxpr))


########################################
##### Flax Utilities
########################################


def get_metrics(device_metrics):
    """
    This function is similar to flax/training/common_utils.py, but works for
    DistributedArray in alpa.
    """
    # pylint: disable=import-outside-toplevel
    from alpa.device_mesh import prefetch

    prefetch(device_metrics)
    return stack_forest(device_metrics)


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
        for _ in range(number):
            run_func()
        if sync_func:
            sync_func()
        costs.append(time.time() - tic)

    return np.array(costs) / number


def run_with_timeout(func, args=(), kwargs=None, timeout=None):
    """Run a function with timeout."""
    ret_value = []

    def _target_func():
        ret_value.append(func(*args, **(kwargs or {})))

    t = threading.Thread(target=_target_func)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError

    if not ret_value:
        raise RuntimeError

    return ret_value[0]


########################################
##### Array Conversion
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


def infer_start_pos_and_n_elements(tensor_shape, tensor_slice):
    start_pos = 0
    n_elements = 1
    for dim_len, dim_slice in zip(tensor_shape, tensor_slice):
        start_pos = start_pos * dim_len + dim_slice.start
        n_elements = n_elements * (dim_slice.stop - dim_slice.start)
    return start_pos, n_elements


def infer_offset_and_n_elements(tensor_slice):
    """Calculate the offset and #elements before making NCCL calls.

    This function assumes the slice is a continuous subset of the original
    tensor.
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


# Note: use Python jit instead of CPP jit,
# because CPP jit has bugs on _DeviceArray.
if is_worker:
    FLAGS.experimental_cpp_jit = False


# Note(Hao): this function will be jit-ed into as many versions as the possible
# length of start_indices
@partial(jax.jit, donate_argnums=0, static_argnums=2)
def jax_tensor_set(src_buf, update, start_indices):
    """
    In-place write on a JAX buffer.

    Args:
        src_buf: JAX device array.
        update: JAX device array.
        start_indices (tuple[int]): tuple of integers indicating the starting
        indices.
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
    """List all gpu information by calling nvidia-smi."""
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
    """Get the number of hosts and the number of devices per host for benchmark
    scripts."""
    if args.num_hosts is not None or args.num_devices_per_host is not None:
        assert (args.num_hosts is not None and
                args.num_devices_per_host is not None)
        num_hosts, num_devices_per_host = (args.num_hosts,
                                           args.num_devices_per_host)
    else:
        if hasattr(args, "local") and args.local:
            num_hosts = 1
            if global_config.backend == "gpu":
                num_devices_per_host = list_gpu_info().count("UUID")
            elif global_config.backend == "tpu":
                num_devices_per_host = len(jax.devices("tpu"))
            else:
                raise ValueError(
                    f"Unsupported backend: {global_config.backend}")
        else:
            ray.init(address="auto")
            num_hosts = len(ray.nodes())
            num_devices_per_host = int(
                ray.cluster_resources()["GPU"]) // num_hosts
    return num_hosts, num_devices_per_host


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
##### Ray Compatibilityu API Utilities
########################################


def try_import_ray_worker(error: bool = False):
    """Tries importing `ray.worker` and returns the module (or None).

    Args:
        error: Whether to raise an error if ray.worker cannot be imported.

    Returns:
        The `ray.worker` modules.

    Raises:
        ImportError: If error=True and ray's version >= 2.0.
    """
    # In the ray-nightly version,
    # worker = _DeprecationWrapper("worker", ray._private.worker)
    # `_DeprecationWrapper` has attributes of `_real_worker`
    try:
        if hasattr(ray.worker, "_real_worker"):
            if error:
                raise ImportError("Could not import `ray.worker`!"
                                  "You might use the ray-nightly "
                                  "and `ray.worker` is deprecated there"
                                  "`pip install ray==1.13.0`.")
            return ray.worker._real_worker  # pylint: disable=protected-access
        else:
            return ray.worker
    except ModuleNotFoundError:
        return ray._private.worker  # pylint: disable=protected-access


def try_import_ray_state(error: bool = False):
    """Tries importing `ray.state` and returns the module (or None).

    Args:
        error: Whether to raise an error if ray.state cannot be imported.

    Returns:
        The `ray.state` modules.

    Raises:
        ImportError: If error=True and ray's version >= 2.0.
    """
    # In the ray-nightly version,
    # state = _DeprecationWrapper("state", ray._private.state)
    # `_DeprecationWrapper` has attributes of `_real_worker`
    try:
        if hasattr(ray.state, "_real_worker"):
            if error:
                raise ImportError("Could not import `ray.state`!"
                                  "You might use the ray-nightly "
                                  "and `ray.state` is deprecated there"
                                  "`pip install ray>=1.13.0`.")
            return ray.state._real_worker  # pylint: disable=protected-access
        else:
            return ray.state
    except ModuleNotFoundError:
        return ray._private.state  # pylint: disable=protected-access


########################################
##### Ray Palcement Group API Utilities
########################################


def is_ray_node_resource(resource_key):
    """Check if the current resource is the host ip."""
    ishost_regex = re.compile(r"^node:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    return ishost_regex.match(resource_key)


def get_bundle2ip(pg: PlacementGroup = None):
    """get the ip address list from placement group

    The ordering of the ip address are aligned with each bundle index.
    """

    if pg:
        pg_id = pg.id.hex()
    # dictionary: bundle_group to node_ip
    dict_bg2ip = {}

    ray_state = try_import_ray_state()
    resources_list = ray_state.state._available_resources_per_node(  # pylint: disable=protected-access
    ).values()

    for resource in resources_list:
        resource_name_list = resource.keys()

        node_ip = None
        bundle_index_list = []
        for resource_name in resource_name_list:
            # when bundles are created, pg resources are
            # specified as [resource]_[bundle_index]_[pg_id]
            if pg:
                try_bundle_index = re.findall(rf"bundle_group_(\d+)_{pg_id}",
                                              resource_name)
            else:
                try_bundle_index = re.findall(r"bundle_group_(\d+)_.*",
                                              resource_name)

            try_node_ip = re.findall(
                r"^node:(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$)", resource_name)

            if try_node_ip:
                node_ip = try_node_ip[0]

            if try_bundle_index:
                bundle_index_list.append(try_bundle_index[0])

        dict_bg2ip.update(
            **dict(zip(bundle_index_list, [node_ip] * len(bundle_index_list))))

    ip_list = []
    for i in range(len(dict_bg2ip)):
        ip_list.append(dict_bg2ip[str(i)])

    return ip_list


def env_integer(key, default):
    if key in os.environ:
        value = os.environ[key]
        if value.isdigit():
            return int(os.environ[key])

        logger.debug(f"Found {key} in environment, but value must "
                     f"be an integer. Got: {value}. Returning "
                     f"provided default {default}.")
        return default
    return default


def create_placement_group(num_hosts,
                           host_num_devices,
                           name,
                           additional_resources_per_host=None):
    """Creates a placement group if it does not exist.

    If a placement group is already detected (in Tune integration),
    this will be a no-op.

    By default the placement group will be created with `SPREAD` strategy.
    This is optimized for colocating GPUs on different nodes.

    Args:
        num_hosts: the number of hosts to create the placement group for
        host_num_devices: the number of devices on each host
        additional_resources_per_host: additional resources per host

    Returns:
        The placement group
    """
    current_placement_group = get_current_placement_group()
    ray_worker = try_import_ray_worker()
    worker = ray_worker.global_worker  # pylint: disable=protected-access
    should_capture_child_tasks_in_placement_group = (
        worker.should_capture_child_tasks_in_placement_group)
    should_create_placement_group = (
        current_placement_group is None or
        not should_capture_child_tasks_in_placement_group)

    if should_create_placement_group:
        # `should_create_placement_group` is always True when using alpa alone.
        # `should_create_placement_group` can be false when integrated with Tune
        additional_resources_per_host = (additional_resources_per_host or {})
        bundles = [{
            "CPU": 1,
            "GPU": host_num_devices[i],
            **additional_resources_per_host
        } for i in range(num_hosts)]

        # Alpa Placement Group: `SPREAD` strategy is required
        # https://docs.ray.io/en/latest/ray-core/placement-group.html#strategy-types
        # Each bundle must be scheduled in a separate node.
        strategy = "SPREAD"

        placement_group = ray.util.placement_group(bundles,
                                                   strategy=strategy,
                                                   name=name or "")
        logger.debug("Waiting for placement group to start.")
        timeout = env_integer(PLACEMENT_GROUP_TIMEOUT_S_ENV, 100)
        ready, _ = ray.wait([placement_group.ready()], timeout=timeout)
        if ready:
            logger.debug("Placement group has started.")
        else:
            raise TimeoutError(
                "Placement group creation timed out. Make sure your "
                "cluster either has enough resources or use an "
                "autoscaling cluster. If you are running on a cluster, "
                "make sure you specify an address in `ray.init()`, for example,"
                ' `ray.init("auto")`. You can also increase the timeout by '
                "setting the ALPA_PLACEMENT_GROUP_TIMEOUT_S environment "
                "variable. Current resources available: "
                f"{ray.available_resources()}, resources requested by "
                f"the placement group: {placement_group.bundle_specs}")
        return placement_group
    else:
        return current_placement_group


def get_bundle_idx(placement_group: PlacementGroup, node_ips: List[str]):
    """Get the bundle index for the placement group.

    The placement group is a list of resource bundles.
    Each bundle will be assigned to **one** node.

    First, we need to find the bundle index with GPU resources.
    Then, we can find the node IP for the bundle index.
    Lastly, we sort bundle index according to the node IP list given.

    Args:
        placement_group: The placement group.
        node_ips: The list of node IP addresses.

    Returns:
        list: The sorted bundle index list.
    """
    # get the node IP for the bundle index
    bundle_ips = get_bundle2ip(placement_group)
    bundle_specs = placement_group.bundle_specs

    # filter out the bundle index with node (GPUs)
    node_bundle_idx_list = [
        i for i, bundle_spec in enumerate(bundle_specs)
        if bundle_spec.get("GPU", 0) > 0
    ]

    if len(node_bundle_idx_list) < len(node_ips):
        raise ValueError("The number of bundles with GPU resources "
                         "is less than the number of node IPs.")

    # node IP -> bundle index
    bundle_ip2idx = {bundle_ips[i]: i for i in node_bundle_idx_list}

    # sorted bundle index according to the node IP list given
    sorted_bundle_idx = [bundle_ip2idx[ip] for ip in node_ips]

    return sorted_bundle_idx


def retrieve_placement_group():
    """retrieve the placement group to support node affinity scheduling

    If already inside the placement group, retrieve the current placement
    group (case I). Then, if the placement group is detected globally in
    alpa, retrieve the global placement group (case II).

    """
    # case 1:
    # Get the current placement group which a task or actor is using
    current_placement_group = get_current_placement_group()
    if current_placement_group:
        return current_placement_group

    # case 2:
    # Get the placement group created when alpa.init('ray')
    global_cluster = device_mesh.global_cluster
    if global_cluster and global_cluster.placement_group:
        alpa_placement_group = global_cluster.placement_group
        return alpa_placement_group

    raise ValueError(
        "The alpa training is not inside the ray tasks or actor or "
        "the placement group is not created yet. One reason is that "
        "Alpa is not connected to Ray cluster, and use `alpa.init('ray')`"
        " at the beginning. Do you have override the placement group? "
        "If not, please help file an issue on Github.")


def get_num_available_gpus(pg: PlacementGroup):
    res = ray.available_resources()
    pg_id = pg.id.hex()
    return res[f"GPU_group_{pg_id}"]


########################################
##### Other Utilities
########################################

GB = 1 << 30  # Gigabyte
MB = 1 << 20  # Megabyte


def map_to_shape(array_pytree: PyTreeDef):
    """Map a PyTree of jax arrays to their shapes."""
    return tree_map(lambda x: getattr(x, "shape", None), array_pytree)


def map_to_nparray(tree: PyTreeDef):
    """Map a PyTree to a PyTree of numpy array."""

    def convert_to_nparray(x):
        if hasattr(x, "__array__"):
            return np.asarray(x)
        return x

    return jax.tree_map(convert_to_nparray, tree)


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


def compute_gpt_tflops(batch_size,
                       seq_len,
                       num_layers,
                       hidden_size,
                       vocab_size,
                       num_gpus,
                       latency,
                       backward=True,
                       checkpoint_activations=False):
    """
    Compute the Tera Flop Operations (TFLOP) per second per GPU
    for GPT-like models.
    """
    factor = 24
    if backward:
        factor += 48
    if checkpoint_activations:
        factor += 24

    total_flop = (factor * batch_size * seq_len *
                  (hidden_size**2) * num_layers * (1 + seq_len /
                                                   (6 * hidden_size)) +
                  6 * batch_size * seq_len * hidden_size * vocab_size)
    # Note: The above formula does not count the first embedding table lookup
    # because it is a sparse operation.
    # If we use dense dot to compute the first embedding table lookup,
    # then the last term in total_flops should be
    # "+ 10 * batch_size * seq_len * hidden_size * vocab_size".
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops


_DISABLE_NUMBA = False


def maybe_numba_jit(func):
    """Decorator to mark a function as numba jitted if numba is available."""
    try:
        from numba import jit  # pylint: disable=import-outside-toplevel
        jitted_func = jit(nopython=True)(func)

        def wrapper(*args, **kwargs):
            if _DISABLE_NUMBA:
                return func(*args, **kwargs)
            return jitted_func(*args, **kwargs)

        return wrapper
    except ImportError:
        logger.warning("Install numba to jit and accelerate the function.")
        return func

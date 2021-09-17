# pylint: disable=consider-using-enumerate
"""Common utilities."""
from collections import OrderedDict
from functools import partial
import itertools as it
import os
import subprocess
import time
from typing import Sequence
from warnings import warn

import cupy as cp
import flax
from flax.training import train_state
import jax
from jax._src.api import FLAGS
from jax._src.dlpack import from_dlpack
from jax.api_util import shaped_abstractify
from jax.core import ClosedJaxpr, DropVar, Jaxpr, Literal, ShapedArray, Var
from jax.experimental.maps import FrozenDict
from jax.interpreters import xla, pxla
from jax.interpreters.xla import _DeviceArray
from jax.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
from jax.tree_util import tree_map, tree_flatten
import numpy as np

# Note: use Python jit instead of CPP jit,
# because CPP jit has bugs on _DeviceArray.
FLAGS.experimental_cpp_jit = False

########################################
##### Parax API Utilities
########################################


def freeze_dict(pytree):
    """Convert a pytree to a FrozenDict."""

    def is_leaf(x):
        return isinstance(x, dict)

    def freeze(x):
        if isinstance(x, dict):
            return FrozenDict(x)
        return x

    return tree_map(freeze, pytree, is_leaf)


def auto_static_argnums(args):
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

    return [i for i in range(len(args)) if is_static_arg(args[i])]


def auto_donate_argnums(args):
    """Return the indices of donated arguments according to heuristic rules."""

    def should_donate(x):
        # Always donate optimizer
        if isinstance(x, (flax.optim.base.Optimizer, train_state.TrainState)):
            return True
        return False

    return [i for i in range(len(args)) if should_donate(args[i])]


########################################
##### Data Structure Utilities
########################################


def to_int_tuple(array):
    """Convert a numpy array to int tuple."""
    if array is None:
        return tuple()
    return tuple(int(x) for x in array)


def get_dim_last_value(array, dim):
    """Get the value of the last element in a dimension."""
    indices = tuple(0 if i != dim else array.shape[dim] - 1
                    for i in range(len(array.shape)))
    return array[indices]


class FastLookupList:

    def __init__(self, iterable=()):
        self.elements = list(iterable)
        self.elements_set = set(iterable)

    def __getitem__(self, key):
        return self.elements[key]

    def __len__(self):
        return len(self.elements)

    def __contains__(self, element):
        return element in self.elements_set

    def append(self, element):
        self.elements.append(element)
        self.elements_set.add(element)


class OrderedSet:

    def __init__(self):
        self.dict = OrderedDict()

    def add(self, *args):
        for x in args:
            self.dict[x] = None

    def update(self, container):
        for x in container:
            self.dict[x] = None

    def __iter__(self):
        for x in self.dict:
            yield x


########################################
##### XLA API Utilities
########################################


def get_compile_options(num_replicas, num_partitions, device_assignment,
                        use_spmd_partitioning, parameter_is_tupled_arguments,
                        build_random_seed):
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


def jaxpr_to_hlo_computation(name, closed_jaxpr, donated_invars, backend):
    """Convert a jaxpr to a XLA HLO computation."""
    backend_name = backend.platform
    in_avals = [var.aval for var in closed_jaxpr.jaxpr.invars]
    consts = closed_jaxpr.consts
    map(xla.prefetch, it.chain(consts, xla.jaxpr_literals(closed_jaxpr.jaxpr)))

    # Convert jaxpr to XLA HLO
    tuple_args = False
    c = xb.make_computation_builder(name)
    xla_consts = xla._xla_consts(c, consts)
    xla_args, donated_invars = xla._xla_callable_args(
        c, in_avals, tuple_args, donated_invars=donated_invars)
    axis_env = xla.AxisEnv(nreps=1, names=(),
                           sizes=())  # All named axes have been vmapped
    out_nodes = xla.jaxpr_subcomp(c, closed_jaxpr.jaxpr, backend_name, axis_env,
                                  xla_consts, name, *xla_args)
    out_tuple = xc.ops.Tuple(c, out_nodes)

    # Set up aliases (donating invars)
    if donated_invars:
        if backend.platform in ("gpu", "tpu"):
            donation_results = xla.set_up_aliases(c, xla_args, out_tuple,
                                                  donated_invars, tuple_args)
        if any(donation_results):
            unused_donations = [
                str(c.GetShape(a))
                for a, d in zip(xla_args, donation_results)
                if d
            ]
            warn("Some donated buffers were not usable: {}".format(
                ", ".join(unused_donations)))

    built = c.build(out_tuple)
    return built


def setup_computation_alias(xla_computation, donated_invars: Sequence[bool]):
    """Set input/output alias in xla computation.

    Assume the tensors in output tuple strictly match the donated parameters.
    """
    program_shape = xla_computation.program_shape()
    parameter_shapes = program_shape.parameter_shapes()
    result_shapes = program_shape.result_shape().tuple_shapes()

    assert len(parameter_shapes) == len(donated_invars)

    ct = 0
    for i in range(len(parameter_shapes)):
        if donated_invars[i]:
            assert parameter_shapes[i].dimensions(
            ) == result_shapes[ct].dimensions()
            xla_computation.setup_alias((ct,), i, ())
            ct += 1


def count_communication_primitives(hlo_ir, ignore_scalar_all_reduce=False):
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


def compile_allocate_zero_buffers(backend, num_devices, shapes, dtypes):
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


def get_shard_shape(aval, sharding_spec):
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
##### Profiling Utilities
########################################


def profile_xla_executable(compiled, backend, local_devices):
    """Measure the time costs of a xla executable with dummy inputs."""
    hlo_module = compiled.hlo_modules()[0]

    # Allocate dummy buffers
    input_shapes = hlo_module.parameter_shapes()
    device_inputs = []
    for shape in input_shapes:
        device_inputs.append([
            backend.buffer_from_pyval(
                np.empty(shape.dimensions(), shape.numpy_dtype()),
                local_devices[i]) for i in range(len(local_devices))
        ])

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

    costs = benchmark_func(run_func, repeat=3, number=3)
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
    the avarage execution time of `number` executions.

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


def xla_buffer_to_jax_buffer(xla_buf):
    """
    Convert an xla buffer to a JAX DeviceArray.

    So we can index over the data buffer.
    """
    aval = ShapedArray(xla_buf.shape, xla_buf.dtype)
    return _DeviceArray(aval, xla_buf.device(), xla_buf)


def jax_buffer_to_xla_buffer(jax_buf):
    """Convert a JAX Device array back to XLA buffer."""
    return jax_buf.device_buffer


# Note(Hao): this function will be jit-ed into as many versions as the possible length of start_indices
@partial(jax.jit, donate_argnums=0, static_argnums=2)
def jax_buffer_set(src_buf, update, start_indices):
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


def to_cupy(tensors):
    """Convert a Jax DeviceArray to cupy tensor; zero copy."""
    if isinstance(tensors, list):
        return list(map(to_cupy, tensors))
    ctensor = cp.fromDlpack(get_jax_dlpack(tensors))
    return ctensor


def to_jax_tensor(tensor):
    """Convert cupy tensors to JAX tensors."""
    if isinstance(tensor, list):
        return list(map(to_jax_tensor, tensor))
    return from_dlpack(tensor.toDlpack())


def get_jax_dlpack(tensor):
    """Helper function for calling dlpack in JAX."""
    return xc._xla.buffer_to_dlpack_managed_tensor(tensor.device_buffer,
                                                   take_ownership=False)


########################################
##### OS / IO Utilities
########################################


def run_cmd(cmd):
    """Run a bash commond."""
    print(cmd)
    os.system(cmd)


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


def write_tsv(heads, values, filename, print_line=True):
    """Write tsv data to a file."""
    assert len(heads) == len(values)

    with open(filename, "a") as fout:
        fout.write("\t".join(values) + "\n")

    if print_line:
        line = ""
        for i in range(len(heads)):
            line += heads[i] + ": " + values[i] + "  "
        print(line)


########################################
##### Other Utilities
########################################

GB = 1 << 30  # Gigabyte
MB = 1 << 20  # Megabyte


def map_to_shape(array_pytree):
    """Map a PyTree of jax arrays to their shapes."""
    return tree_map(lambda x: getattr(x, "shape", None), array_pytree)


def compute_bytes(pytree):
    """Compute the total bytes of arrays in a pytree."""
    flatten_args, _ = tree_flatten(pytree)
    ret = 0
    for x in flatten_args:
        if hasattr(x, "shape"):
            ret += np.prod(x.shape) * x.dtype.itemsize
    return ret


def get_micro_batch(batch_invars, num_micro_batches, *raw_avals):
    avals = []
    for aval, is_batch_var in zip(raw_avals, batch_invars):
        if is_batch_var:
            assert aval.shape[0] % num_micro_batches == 0,\
                "The batch dimension must be divisable by num_micro_batches."
            shape = (aval.shape[0] // num_micro_batches,) + aval.shape[1:]
            avals.append(aval.update(shape=shape))
        else:
            avals.append(aval)
    return avals


def slices_to_jaxpr(closed_jaxpr: ClosedJaxpr,
                    sliced_eqns) -> Sequence[ClosedJaxpr]:
    N = len(sliced_eqns)
    global_invars = set(closed_jaxpr.jaxpr.invars)
    global_consts = dict(zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))
    global_outvars = set(
        var for var in closed_jaxpr.jaxpr.outvars if isinstance(var, Var))
    result = []
    layer_invars = [set() for _ in range(N)]
    layer_outvars = [set() for _ in range(N)]
    layer_consts = [dict() for _ in range(N)]
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

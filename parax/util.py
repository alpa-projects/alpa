"""Common utilities."""
import os
import subprocess
import time

import flax
import numpy as np
from jax.api_util import shaped_abstractify
from jax.experimental.maps import FrozenDict
from jax.lib import xla_bridge as xb
from jax.tree_util import tree_map, tree_flatten


########################################
##### API Utilities
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

        if isinstance(arg, flax.optim.base.Optimizer):
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
        if isinstance(x, flax.optim.base.Optimizer):
            return True
        return False

    return [i for i in range(len(args)) if should_donate(args[i])]


########################################
##### Data Structure Utilities
########################################

def to_int_tuple(array):
    """Convert a numpy array to int tuple."""
    return tuple(int(x) for x in array)


def get_dim_last_value(array, dim):
    """Get the value of the last element in a dimension."""
    indices = tuple(0 if i != dim else array.shape[dim] - 1 for i in range(len(array.shape)))
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


########################################
##### XLA API Utilities
########################################

def get_compile_options(num_replicas,
                        num_partitions,
                        device_assignment,
                        use_spmd_partitioning,
                        parameter_is_tupled_arguments,
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

########################################
##### Profiling Utilities
########################################

def profile_xla_executable(compiled, backend, local_devices, sync_func):
    """Measure the time costs of a xla executable."""
    hlo_module = compiled.hlo_modules()[0]
    input_shapes = hlo_module.parameter_shapes()
    device_inputs = []
    for shape in input_shapes:
        device_inputs.append(
            [backend.buffer_from_pyval(np.empty(shape.dimensions(), shape.numpy_dtype()),
                                       local_devices[i])
             for i in range(len(local_devices))]
        )

    def run_once():
        device_outputs = compiled.execute_sharded_on_local_devices(device_inputs)

        # Reset the value for donate buffers
        for j in range(len(device_inputs)):
            if device_inputs[j][0].is_deleted():
                device_inputs[j] = device_outputs[j]
        sync_func()

    costs = measure_func(run_once, repeat=1, min_repeat_second=0.5)
    return costs


def measure_func(func, warmup=1, number=10, repeat=3, min_repeat_second=0):
    """Measure the execution time of a function.

    The function is executed for (warmup + number * repeat) times.
    The return value is a array of `repeat` elements and each elements is 
    the avarage execution time of `number` executions.

    If `min_repeat_second` is set, the function automatically picks a `number`
    so that one `repeat` lasts for at least `min_repeat_second` seconds.
    """
    for i in range(warmup):
        func()

    if min_repeat_second:
        tic = time.time()
        func()
        toc = time.time()
        cost = toc - tic
        number = max(int(min_repeat_second / cost), 1)

    costs = []
    for i in range(repeat):
        tic = time.time()
        for j in range(number):
            func()
        toc = time.time()
        costs.append((toc - tic) / number)

    return costs


########################################
##### Other Utilities
########################################

GB = 1 << 30  # Gigabyte
MB = 1 << 20  # Megabyte


def run_cmd(cmd):
    """Run a bash commond."""
    print(cmd)
    os.system(cmd)


def compute_bytes(pytree):
    """Compute the total bytes of arrays in a pytree."""
    flatten_args, _ = tree_flatten(pytree)
    ret = 0
    for x in flatten_args:
        if hasattr(x, "shape"):
            ret += np.prod(x.shape) * x.dtype.itemsize
    return ret


def list_gpu_info():
    """List all gpu information by calling nvidia-sim."""
    ret = subprocess.getoutput("nvidia-smi -L")
    return ret


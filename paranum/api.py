from functools import wraps

import jax

def process_batch(batch, n_devices):
    def split(x):
        assert x.shape[0] % n_devices == 0
        return x.reshape((n_devices, x.shape[0] // n_devices) + x.shape[1:])
    return jax.tree_util.tree_map(split, batch)


def parallelize(func):
    n_devices = len(jax.devices())

    func_parallel = jax.pmap(jax.jit(func), axis_name='auto_parallel_batch',
        in_axes=(None, 0), out_axes=None)

    @wraps(func_parallel)
    def ret_func(weights, batch):
        batch = process_batch(batch, n_devices)
        return func_parallel(weights, batch)

    return ret_func


def annotate_gradient(gradients):
    from jax.core import thread_local_state

    axis_env = thread_local_state.trace_state.axis_env
    in_auto_parallel = False
    for x in axis_env:
        if x.name == 'auto_parallel_batch':
            in_auto_parallel = True

    if in_auto_parallel:
        return jax.lax.pmean(gradients, 'auto_parallel_batch')
    else:
        return gradients


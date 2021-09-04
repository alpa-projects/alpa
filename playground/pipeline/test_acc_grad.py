import jax
from jax import jit, grad, tree_flatten
from jax._src.api import make_jaxpr
from jax.core import DropVar, jaxpr_as_fun, gensym
import jax.numpy as jnp

import parax
from parax.pipeline_parallel.manual_pipeline import manual_pipeline
from parax.pipeline_parallel.stage import (
    compute_to_acc_pipe, rewrite_apply_grad,
    slice_closed_jaxpr_by_manual_pipeline_marks, mark_global_and_local_vars,
    mark_grad_mesh, slice_apply_gradient)
from parax.pipeline_parallel.three_d_parallel import split_compute_and_apply
from parax.pipeline_parallel.primitive_def import mark_pipeline

from flax import linen as nn, optim


class MLP_Model(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        # FIXME (zhuohan): if don't require the gradient of x here, the
        #                  backward pass of the pipeline start will not
        #                  be generated.
        mark_pipeline(name='1', mark_type='start')
        x = nn.Dense(features=self.hidden_dim, use_bias=False)(x)
        x = nn.relu(x)
        mark_pipeline(name='1', mark_type='end')
        mark_pipeline(name='2', mark_type='start')
        x = nn.Dense(features=self.output_dim, use_bias=False)(x)
        return x


batch_size = 128
hidden_dim = 2048
input_dim = output_dim = hidden_dim
model = MLP_Model(hidden_dim=hidden_dim, output_dim=output_dim)
x = jnp.ones((batch_size, input_dim))
y = jnp.ones((batch_size, output_dim))
rngkey = jax.random.PRNGKey(0)
params = model.init(rngkey, x)
optimizer = optim.GradientDescent(1e-2).create(params)


@manual_pipeline
def loss_func(params, x, y):
    out = model.apply(params, x)
    loss = jnp.mean((out - y)**2)
    mark_pipeline(name='2', mark_type='end')
    return loss


def test_compute_to_accumulate():
    compute_grad = grad(loss_func, argnums=(0, 1, 2))
    params = optimizer.target
    compute_grad_jaxpr = make_jaxpr(compute_grad)(params, x, y)
    flatten_args, _ = tree_flatten((params, x, y))
    acc_grad_jaxpr, grad_outs = compute_to_acc_pipe(compute_grad_jaxpr)
    grad_zeros = [jnp.zeros_like(val) for val in acc_grad_jaxpr.out_avals]
    # donate_argnums = [
    #     i for i in range(len(donated_invars)) if donated_invars[i]
    # ]
    args = params, x, y
    new_args = flatten_args + grad_zeros
    jitted_fn = jit(jaxpr_as_fun(acc_grad_jaxpr))
    outs = jitted_fn(*new_args)

    new_args = flatten_args + list(outs)
    double_outs = jitted_fn(*new_args)

    correct = map(lambda x: 2 * x, tree_flatten(compute_grad(*args))[0])
    for test, corr in zip(double_outs, correct):
        assert jnp.allclose(test, corr)


def get_invals_from_env(closed_jaxpr, env):
    vars = closed_jaxpr.jaxpr.invars
    return [env[var] for var in vars]


def get_vals_from_env(vars, env):
    return [env[var] for var in vars]


def record_values(vars, avals, env):
    for var, aval in zip(vars, avals):
        if isinstance(var, DropVar):
            continue
        if var in env:
            assert jnp.allclose(env[var], aval)
        env[var] = aval


def get_and_set(closed_jaxpr, env):
    outs = jaxpr_as_fun(closed_jaxpr)(
        *get_invals_from_env(closed_jaxpr, env)
    )
    record_values(closed_jaxpr.jaxpr.outvars, outs, env)


def test_compute_and_apply():
    def train_step(optimizer, batch):
        grad_param, _x, _y = parax.grad(loss_func,
                                        argnums=(0, 1, 2))(optimizer.target,
                                                           batch['x'],
                                                           batch['y'])
        new_optimizer = optimizer.apply_gradient(grad_param)
        return new_optimizer

    batch = {"x": x, "y": y}
    closed_jaxpr = make_jaxpr(train_step)(optimizer, batch)
    compute_grad_jaxpr, old_apply_grad_jaxpr, barrier = split_compute_and_apply(
        closed_jaxpr)
    # compute grad to accumulate grad
    acc_grad_jaxpr, acc_grad_dict = compute_to_acc_pipe(compute_grad_jaxpr)
    # change invars of apply grad to output of accumulate grad
    apply_grad_jaxpr = rewrite_apply_grad(old_apply_grad_jaxpr, acc_grad_dict,
                                          barrier)
    # slice accumulate grad
    gensym_func = gensym([closed_jaxpr.jaxpr])
    jax_pipeline_stages = slice_closed_jaxpr_by_manual_pipeline_marks(
        acc_grad_jaxpr)
    jax_pipeline_stages = [
        mark_global_and_local_vars(stage, gensym_func)
        for stage in jax_pipeline_stages
    ]
    # delete the two lines below in auto mesh version
    stage_num = len(jax_pipeline_stages)
    stage_to_mesh = {
        i: (i if i < stage_num / 2 else stage_num - i - 1)
        for i, _ in enumerate(jax_pipeline_stages)
    }
    # slice apply-grad stages
    grad_mesh = mark_grad_mesh(apply_grad_jaxpr.jaxpr.invars,
                               jax_pipeline_stages, stage_to_mesh)
    sliced_apply_grad, _ = slice_apply_gradient(apply_grad_jaxpr, grad_mesh)

    # Simulation:
    # correct result:
    args, _ = tree_flatten((optimizer, batch))
    env = dict()
    record_values(closed_jaxpr.jaxpr.invars, args, env)
    correct = jaxpr_as_fun(closed_jaxpr)(
        *get_invals_from_env(closed_jaxpr, env))
    # Test 1: split compute and apply
    from copy import copy
    env_1 = copy(env)
    get_and_set(compute_grad_jaxpr, env_1)
    for inv, outv in zip(barrier.invars, barrier.outvars):
        if not isinstance(outv, DropVar) and inv in env_1:
            env_1[outv] = env_1[inv]
    get_and_set(old_apply_grad_jaxpr, env_1)
    outs = get_vals_from_env(closed_jaxpr.jaxpr.outvars, env_1)
    for t, c in zip(outs, correct):
        assert jnp.allclose(t, c)
    # Test 2: accumulate and apply
    env_2 = copy(env)
    grad_num = len(acc_grad_jaxpr.out_avals)
    grad_invars = set(acc_grad_jaxpr.jaxpr.invars[-1 * grad_num:])
    for inv in acc_grad_jaxpr.jaxpr.invars:
        if inv not in env_2:
            assert inv in grad_invars
            env_2[inv] = jnp.zeros_like(inv.aval)
    get_and_set(acc_grad_jaxpr, env_2)
    get_and_set(apply_grad_jaxpr, env_2)
    outs = get_vals_from_env(closed_jaxpr.jaxpr.outvars, env_2)
    for t, c in zip(outs, correct):
        assert jnp.allclose(t, c)
    return

test_compute_to_accumulate()
test_compute_and_apply()
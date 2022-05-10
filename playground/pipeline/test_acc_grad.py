import jax
from jax import jit, grad, tree_flatten
from jax._src.api import make_jaxpr
from jax.core import DropVar, jaxpr_as_fun, gensym
import jax.numpy as jnp
import numpy as np

import alpa
from alpa.pipeline_parallel.manual_layer_slicing import manual_layer_slicing
from alpa.pipeline_parallel.computation import (
    apply_grad_add_marker, compute_grad_to_accumulate_grad, apply_grad_get_mean,
    get_var_mapping, slice_closed_jaxpr_by_full_pipeline_marks,
    mark_missing_vars_in_backward_computation_pipeline_marks, mark_gradvar_to_mesh, slice_apply_gradient,
    replace_all_with)
from alpa.pipeline_parallel.three_d_parallel import split_compute_grad_and_apply_grad, split_donate_invars
from alpa.pipeline_parallel.primitive_def import mark_pipeline

from flax import linen as nn, optim

from copy import copy


class MLP_Model(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        mark_pipeline(name='1', mark_type='start')
        x = nn.Dense(features=self.hidden_dim, use_bias=False)(x)
        x = nn.relu(x)
        mark_pipeline(name='1', mark_type='end')
        mark_pipeline(name='2', mark_type='start')
        x = nn.Dense(features=self.output_dim, use_bias=False)(x)
        return x


batch_size = 4
hidden_dim = 3
input_dim = output_dim = hidden_dim
model = MLP_Model(hidden_dim=hidden_dim, output_dim=output_dim)
x = jnp.array(np.random.rand(batch_size, output_dim))
y = jnp.array(np.random.rand(batch_size, output_dim))
rngkey = jax.random.PRNGKey(0)
params = model.init(rngkey, x)
optimizer = optim.GradientDescent(1e-2).create(params)
batch = {"x": x, "y": y}
grad_in_to_out = None


@manual_layer_slicing
def loss_func(params, x, y):
    out = model.apply(params, x)
    loss = jnp.mean((out - y)**2)
    mark_pipeline(name='2', mark_type='end')
    return loss


def train_step(optimizer, batch):
    grad_param, _x, _y = alpa.grad(loss_func,
                                    argnums=(0, 1, 2))(optimizer.target,
                                                       batch['x'], batch['y'])
    new_optimizer = optimizer.apply_gradient(grad_param)
    return new_optimizer


def test_compute_to_accumulate():
    compute_grad = grad(loss_func, argnums=(0, 1, 2))
    params = optimizer.target
    compute_grad_jaxpr = make_jaxpr(compute_grad)(params, x, y)
    gensym_fn = gensym([compute_grad_jaxpr.jaxpr])
    flatten_args, _ = tree_flatten((params, x, y))
    reduction_vector = [True] * len(compute_grad_jaxpr.jaxpr.outvars)
    acc_grad_jaxpr, grad_outs, _ = compute_grad_to_accumulate_grad(compute_grad_jaxpr,
                                                                   reduction_vector,
                                                                   gensym_fn)
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


def get_invals_from_env(closed_jaxpr, env, batch_num=0):
    vars = closed_jaxpr.jaxpr.invars
    if batch_num == 0:
        return [env[batch_num][repr(var)] for var in vars]
    vals = []
    for var in vars:
        if var in grad_in_to_out:
            vals.append(env[batch_num - 1][grad_in_to_out[var]])
        else:
            vals.append(env[batch_num][repr(var)])
    return vals


def get_vals_from_env(vars, env, batch_num=0):
    return [env[batch_num][repr(var)] for var in vars]


def record_values(vars, avals, env, batch_num=0):
    for var, aval in zip(vars, avals):
        if isinstance(var, DropVar):
            continue
        key = repr(var)
        if key in env[batch_num]:
            assert jnp.allclose(env[batch_num][key], aval)
        env[batch_num][key] = aval


def get_and_set(closed_jaxpr, env, batch_num=0, donate_argnums=()):
    outs = jax.jit(jaxpr_as_fun(closed_jaxpr), donate_argnums=donate_argnums)(
        *get_invals_from_env(closed_jaxpr, env, batch_num))
    record_values(closed_jaxpr.jaxpr.outvars, outs, env, batch_num)


def test_compute_and_apply_basic():
    closed_jaxpr = make_jaxpr(train_step)(optimizer, batch)
    gensym_func = gensym([closed_jaxpr.jaxpr])
    compute_grad_jaxpr, old_apply_grad_jaxpr, barrier = split_compute_grad_and_apply_grad(
        closed_jaxpr)
    # compute grad to accumulate grad
    reduction_vector = [True] * len(compute_grad_jaxpr.jaxpr.outvars)
    acc_grad_jaxpr, acc_grad_dict, _ = compute_grad_to_accumulate_grad(
        compute_grad_jaxpr, reduction_vector, gensym_func)
    # apply-grad
    mask = {
        outv: acc_grad_dict[inv]
        for outv, inv in zip(barrier.outvars, barrier.invars)
        if (not isinstance(outv, DropVar) and
            outv in old_apply_grad_jaxpr.jaxpr.invars)
    }
    # change invars of apply grad to output of accumulate grad
    apply_grad_jaxpr = replace_all_with(old_apply_grad_jaxpr, mask)

    # Simulation:
    # correct result:
    args, _ = tree_flatten((optimizer, batch))
    env = [dict()]
    record_values(closed_jaxpr.jaxpr.invars, args, env)
    correct = jaxpr_as_fun(closed_jaxpr)(
        *get_invals_from_env(closed_jaxpr, env))
    # Test 1: split compute and apply
    env_1 = copy(env)
    get_and_set(compute_grad_jaxpr, env_1)
    for inv, outv in zip(barrier.invars, barrier.outvars):
        if isinstance(outv, DropVar):
            continue
        key = repr(inv)
        if key in env_1[0]:
            env_1[0][repr(outv)] = env_1[0][key]
    get_and_set(old_apply_grad_jaxpr, env_1)
    outs = get_vals_from_env(closed_jaxpr.jaxpr.outvars, env_1)
    for t, c in zip(outs, correct):
        assert jnp.allclose(t, c)
    del env_1
    # Test 2: accumulate and apply
    env_2 = copy(env)
    grad_num = len(acc_grad_jaxpr.out_avals)
    grad_invars = set(acc_grad_jaxpr.jaxpr.invars[-1 * grad_num:])
    for inv in acc_grad_jaxpr.jaxpr.invars:
        key = repr(inv)
        if key not in env_2[0]:
            assert inv in grad_invars
            env_2[0][key] = jnp.zeros_like(inv.aval)
    get_and_set(acc_grad_jaxpr, env_2)
    get_and_set(apply_grad_jaxpr, env_2)
    outs = get_vals_from_env(closed_jaxpr.jaxpr.outvars, env_2)
    for t, c in zip(outs, correct):
        assert jnp.allclose(t, c)


def donate_invars_to_argnums(donate_invars):
    return [i for i, d in enumerate(donate_invars) if d]


def test_compute_and_apply(microbatches):
    closed_jaxpr = make_jaxpr(train_step)(optimizer, batch)
    gensym_func = gensym([closed_jaxpr.jaxpr])
    compute_grad_jaxpr, apply_grad_jaxpr, barrier = split_compute_grad_and_apply_grad(
        closed_jaxpr)
    # compute grad to accumulate grad
    global grad_in_to_out
    reduction_vector = [True] * len(compute_grad_jaxpr.jaxpr.outvars)
    acc_grad_jaxpr, acc_grad_dict, grad_glob_in = compute_grad_to_accumulate_grad(
        compute_grad_jaxpr, reduction_vector, gensym_func)
    grad_in_to_out = grad_glob_in
    # slice accumulate grad
    acc_invars = acc_grad_jaxpr.jaxpr.invars
    acc_outvars = acc_grad_jaxpr.jaxpr.outvars
    jax_pipeline_stages = slice_closed_jaxpr_by_full_pipeline_marks(
        acc_grad_jaxpr)
    jax_pipeline_stages = mark_missing_vars_in_backward_computation_pipeline_marks(
        jax_pipeline_stages, acc_invars, acc_outvars)
    # delete the two lines below in auto mesh version
    stage_num = len(jax_pipeline_stages)
    assert stage_num % 2 == 0
    stage_to_mesh = {
        i: (i if i < stage_num / 2 else stage_num - i - 1)
        for i, _ in enumerate(jax_pipeline_stages)
    }
    mesh_num = int(stage_num / 2)
    # apply-grad
    mask = {
        outv: acc_grad_dict[inv]
        for outv, inv in zip(barrier.outvars, barrier.invars)
        if not isinstance(outv, DropVar)
    }
    # slice apply-grad stages
    global_outvars = closed_jaxpr.jaxpr.outvars
    grad_mesh = mark_gradvar_to_mesh(apply_grad_jaxpr.jaxpr.invars,
                                     jax_pipeline_stages, stage_to_mesh, mask)
    gradients = [g for g in barrier.outvars if not isinstance(g, DropVar)]
    apply_grad_jaxpr, global_outvars = apply_grad_get_mean(apply_grad_jaxpr,
                                                       gradients,
                                                       gensym_func,
                                                       microbatches,
                                                       global_outvars)
    sliced_apply_grad, _ = slice_apply_gradient(apply_grad_jaxpr, grad_mesh,
                                                mesh_num)
    sliced_apply_grad, outvar_map = apply_grad_add_marker(sliced_apply_grad,
                                                          mask,
                                                          gensym_func,
                                                          computation=True)
    global_outvars = list(
        map(lambda x: get_var_mapping(outvar_map, x), global_outvars))
    # donate invars
    donated_invars = (True, True, True, False, False)
    slice_num = len(sliced_apply_grad)
    grad_invars = list(grad_glob_in.keys())
    all_invars = closed_jaxpr.jaxpr.invars + grad_invars
    all_donation = donated_invars + (True,) * len(grad_glob_in)
    jax_all_stages = jax_pipeline_stages + sliced_apply_grad
    # forward, backward and apply gradient is serialized in a batch.
    pattern = [[i, i + slice_num, i + slice_num * 2] for i in range(slice_num)]
    donate_lists = split_donate_invars(all_donation, all_invars, jax_all_stages,
                                       pattern)
    pipe_donate = donate_lists[:slice_num * 2]
    apply_donate = donate_lists[slice_num * 2:]
    # Simulation:
    # correct result:
    args, _ = tree_flatten((optimizer, batch))
    env = [dict()]
    record_values(closed_jaxpr.jaxpr.invars, args, env)
    correct = jaxpr_as_fun(closed_jaxpr)(
        *get_invals_from_env(closed_jaxpr, env))
    # Test 3: slices
    # slices:
    env = [dict() for _ in range(microbatches)]
    non_split_args = tree_flatten(optimizer)[0]
    to_split_args = tree_flatten(batch)[0]
    # this is a rough simulator, so not actually split them but run m times instead
    # split_args = map(lambda x: jnp.split(x, microbatches), to_split_args)
    for b in range(microbatches):
        args = non_split_args + to_split_args
        record_values(closed_jaxpr.jaxpr.invars, args, env, b)
    record_values(closed_jaxpr.jaxpr.invars, args, env)
    env_3 = copy(env)
    grad_num = len(acc_grad_jaxpr.out_avals)
    grad_invars = set(acc_grad_jaxpr.jaxpr.invars[-1 * grad_num:])
    for invar in acc_grad_jaxpr.jaxpr.invars:
        key = repr(invar)
        if key not in env_3[0]:
            assert invar in grad_invars
            env_3[0][key] = jnp.zeros_like(invar.aval)

    for b in range(microbatches):
        for i, stage in enumerate(jax_pipeline_stages):
            get_and_set(stage.closed_jaxpr(), env_3, b)
    # store results of apply grad into microbatches - 1
    for i, stage in enumerate(sliced_apply_grad):
        if stage.outvars:
            get_and_set(stage.closed_jaxpr(), env_3, microbatches - 1)
    outs = get_vals_from_env(global_outvars, env_3, microbatches - 1)
    for t, c in zip(outs, correct):
        assert jnp.allclose(t, c)
    grad_in_to_out = None


test_compute_to_accumulate()
test_compute_and_apply_basic()
test_compute_and_apply(1)
test_compute_and_apply(4)
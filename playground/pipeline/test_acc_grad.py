import jax
from jax import jit, grad, tree_flatten
from jax._src.api import make_jaxpr
from jax.core import jaxpr_as_fun
import jax.numpy as jnp
from parax.pipeline_parallel.manual_pipeline import manual_pipeline

from parax.pipeline_parallel.stage import compute_to_acc_pipe
from parax.pipeline_parallel.primitive_def import mark_pipeline


def test():
    from flax import linen as nn, optim

    class Model(nn.Module):
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
    model = Model(hidden_dim=hidden_dim, output_dim=output_dim)
    x = jnp.ones((batch_size, input_dim))
    y = jnp.ones((batch_size, output_dim))
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, x)
    optimizer = optim.GradientDescent(1e-2).create(params)

    def loss_func(params, x, y):
        out = model.apply(params, x)
        loss = jnp.mean((out - y)**2)
        mark_pipeline(name='2', mark_type='end')
        return loss

    loss_func = manual_pipeline(loss_func)
    compute_grad = grad(loss_func, argnums=(0, 1, 2))
    compute_grad_jaxpr = make_jaxpr(compute_grad)(params, x, y)
    flatten_args, _ = tree_flatten((params, x, y))
    acc_grad_jaxpr, grad_outs = compute_to_acc_pipe(compute_grad_jaxpr)
    grad_len = len(compute_grad_jaxpr.out_avals)
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


test()
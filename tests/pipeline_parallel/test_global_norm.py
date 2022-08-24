import unittest

import jax
from jax import numpy as jnp
from jax.core import gensym
import optax
from optax._src import linear_algebra

from alpa import init, grad, parallelize, PipeshardParallel
from alpa.model.bert_model import BertConfig
from alpa.model.model_util import TrainState
from alpa.pipeline_parallel.apply_grad import (ApplyGradRewriter,
                                               slice_apply_gradient)
from alpa.pipeline_parallel.layer_construction import manual_layer_construction
from alpa.testing import BertLayerModel, assert_allclose
from alpa.util import OrderedSet


class GlobalNormTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def test_split_jaxpr(self):
        # TODO(yonghao): check something to make it a real test
        num_mesh = 2

        def fn(*args):
            ret = 0
            for arg in args:
                ret += jnp.mean(jnp.dot(arg, arg))
            return ret * 10

        args = [jnp.ones((10, 10)) for _ in range(4)]
        jaxpr = jax.make_jaxpr(fn)(*args)

        gensym_fn = gensym([jaxpr.jaxpr])
        invars = jaxpr.jaxpr.invars
        var_mesh = {v: OrderedSet([idx // 2]) for idx, v in enumerate(invars)}
        new_jaxpr = ApplyGradRewriter(jaxpr, var_mesh).split_replicated_eqns(
            gensym_fn, num_mesh)
        # print(jaxpr)
        # print(new_jaxpr)

        grad_mesh = {k: list(v)[0] for k, v in var_mesh.items()}
        outvar_mesh = {}
        jaxprs, _ = slice_apply_gradient(jaxpr, grad_mesh, outvar_mesh,
                                         num_mesh, 2, {})
        # for jaxpr in jaxprs:
        #     print(ApplyGradRewriter.rewrite_allreduce(jaxpr))

    def test_end_to_end(self):

        def train_step(state, batch):

            def loss_func(params, x, y, attention_mask):
                out = state.apply_fn(params, x, attention_mask)
                loss = jnp.mean((out - y)**2)
                return loss

            grads = grad(loss_func)(state.params, batch["x"], batch["y"],
                                    batch["attention_mask"])
            glob_norm = linear_algebra.global_norm(grads)
            return glob_norm

        batch_size = 16
        seq_len = 8
        hidden_size = 128
        num_heads = 8
        dtype = jnp.float32

        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=dtype)
        y = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=dtype)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=dtype)

        # Init model and optimizer
        model = BertLayerModel(config=BertConfig(hidden_size=hidden_size,
                                                 intermediate_size=hidden_size *
                                                 4,
                                                 num_attention_heads=num_heads,
                                                 num_hidden_layers=2))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x, attention_mask)
        tx = optax.sgd(learning_rate=1e-2)
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  tx=tx,
                                  dynamic_scale=None)

        # Train step
        batch = {"x": x, "y": y, "attention_mask": attention_mask}
        gradients = train_step(state, batch)
        method = PipeshardParallel()
        p_train_step = parallelize(train_step, donate_argnums=(), method=method)
        gradients_with_pipeline = p_train_step(state, batch)

        # Check results
        assert_allclose(gradients, gradients_with_pipeline)

def suite():
    suite = unittest.TestSuite()
    # suite.addTest(GlobalNormTest("test_split_jaxpr"))
    suite.addTest(GlobalNormTest("test_end_to_end"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
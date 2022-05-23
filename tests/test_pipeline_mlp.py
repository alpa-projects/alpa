import unittest
import os

import jax
import jax.numpy as jnp
import optax
import ray

from alpa import (init, parallelize, mark_pipeline, manual_layer_construction,
                  PipeshardParallel)
from alpa.parallel_method import LocalPipelineParallel
from alpa.model.model_util import TrainState
from alpa.testing import MLPModel, assert_allclose


class PipelineMLPTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        init(cluster="ray")

    def train_2_layer_mlp(self, method):

        def train_step(state, batch):

            def loss_func(params, x, y):
                out = state.apply_fn(params, x)
                loss = jnp.mean((out - y)**2)
                mark_pipeline(name="2", mark_type="end")
                return loss

            loss_func = manual_layer_construction(loss_func)
            grads = jax.grad(loss_func)(state.params, batch["x"], batch["y"])
            return grads

        batch_size = 64
        hidden_dim = 1024
        input_dim = output_dim = hidden_dim

        x = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, output_dim))

        # Init model and optimizer
        model = MLPModel(hidden_dim=hidden_dim, output_dim=output_dim)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        tx = optax.sgd(learning_rate=1e-2)
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  tx=tx,
                                  dynamic_scale=None)

        # Train step
        batch = {"x": x, "y": y}
        gradients = train_step(state, batch)
        p_train_step = parallelize(train_step, donate_argnums=(), method=method)
        gradients_with_pipeline = p_train_step(state, batch)
        if isinstance(method, PipeshardParallel):
            p_train_step.get_executable(state, batch).get_load_info()

        # Check results
        assert_allclose(gradients, gradients_with_pipeline)

    def test_2_layer_mlp_local_pipeline_parallel(self):
        self.train_2_layer_mlp(LocalPipelineParallel())

    def test_2_layer_mlp_pipeshard_parallel(self):
        self.train_2_layer_mlp(PipeshardParallel())


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineMLPTest("test_2_layer_mlp_local_pipeline_parallel"))
    suite.addTest(PipelineMLPTest("test_2_layer_mlp_pipeshard_parallel"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

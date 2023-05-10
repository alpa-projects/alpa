import unittest

import jax
from jax import numpy as jnp, lax
from jax._src.tree_util import tree_map
from optax import global_norm

from alpa import grad
from alpa.testing import PipelineBasicTest


# FIXME: recover this test suite.
class GlobalNormTest(PipelineBasicTest):

    @unittest.skip("This test is broken. Need to reimplement CrossMeshAllReduce because thunk is being deprecated.")
    def test_global_norm(self):
        hlos = self.run_n_layer_bert(num_layers=2,
                                     manual_pipeline_layer=False,
                                     clip_by_global_norm=True)
        for x in hlos[-2:]:
            assert "CrossMeshAllReduce" in x

    @unittest.skip("No data to test efficiently.")
    def test_dynamic_scale(self):
        hlos = self.run_n_layer_bert(num_layers=2,
                                     manual_pipeline_layer=False,
                                     use_dynamic_scale=True)

    @unittest.skip("No data to test efficiently.")
    def test_global_norm_dynamic_scale(self):
        hlos = self.run_n_layer_bert(num_layers=2,
                                     manual_pipeline_layer=False,
                                     clip_by_global_norm=True,
                                     use_dynamic_scale=True)

    @unittest.skip("This test is broken. Need to reimplement CrossMeshAllReduce because thunk is being deprecated.")
    def test_glob_norm_and_all_le(self):

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"],
                                     batch["attention_mask"])
                loss = jnp.mean((out - batch["y"])**2)
                return loss

            grads = grad(loss_func)(state.params)
            glob_norm = global_norm(grads)
            new_grads = tree_map(lambda g: g / glob_norm, grads)
            new_state = state.apply_gradients(grads=new_grads)

            ls_1 = jnp.array(True)
            for g in jax.tree_util.tree_leaves(grads):
                ls_1 &= jnp.all(lax.le(g, 1.))
            return new_state, (new_grads, ls_1)

        hlos = self.run_n_layer_bert(num_layers=2, inject_train_step=train_step)
        for x in hlos[-2:]:
            assert 'backend_config="SUM;' in x
            assert 'backend_config="AND;' in x
            assert x.count("CrossMeshAllReduce") == 2


def suite():
    suite = unittest.TestSuite()
    suite.addTest(GlobalNormTest("test_global_norm"))
    suite.addTest(GlobalNormTest("test_dynamic_scale"))
    suite.addTest(GlobalNormTest("test_global_norm_dynamic_scale"))
    suite.addTest(GlobalNormTest("test_glob_norm_and_all_le"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

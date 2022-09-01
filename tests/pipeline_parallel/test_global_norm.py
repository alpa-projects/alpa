import unittest

import jax
from jax import numpy as jnp
from jax._src.tree_util import tree_map
from jax.core import gensym

from alpa import init, grad, parallelize, PipeshardParallel
from alpa.model.bert_model import BertConfig
from alpa.model.model_util import TrainState
from alpa.pipeline_parallel.apply_grad import (ApplyGradRewriter,
                                               slice_apply_gradient)
from alpa.testing import PipelineBasicTest
from alpa.util import OrderedSet


class GlobalNormTest(PipelineBasicTest):

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
                                         num_mesh, 2, {}, gensym_fn, False,
                                         [2, 4])
        # for jaxpr in jaxprs:
        #     print(ApplyGradRewriter.rewrite_allreduce(jaxpr))

    def test_global_norm(self):
        self.run_n_layer_bert(num_layers=2, manual_pipeline_layer=False,
                              clip_by_global_norm=True)

    def test_dynamic_scale(self):
        self.run_n_layer_bert(num_layers=2, manual_pipeline_layer=False,
                              use_dynamic_scale=True)

    def test_global_norm_dynamic_scale(self):
        self.run_n_layer_bert(num_layers=2, manual_pipeline_layer=False,
                              clip_by_global_norm=True,
                              use_dynamic_scale=True)


def suite():
    suite = unittest.TestSuite()
    # suite.addTest(GlobalNormTest("test_split_jaxpr"))
    suite.addTest(GlobalNormTest("test_global_norm"))
    #suite.addTest(GlobalNormTest("test_dynamic_scale"))
    #suite.addTest(GlobalNormTest("test_global_norm_dynamic_scale"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

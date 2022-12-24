"""
Test the manual sharding spec.
"""
import itertools
import unittest

import jax
from jax.experimental.pjit import PartitionSpec
from jax.tree_util import tree_map
import jax.numpy as jnp

import alpa
from alpa import (ManualShardingOption, ManualStageOption, PipeshardParallel,
                  mark_pipeline_boundary, parallelize)


class PipeshardManualShardingTest(unittest.TestCase):

    def setUp(self):
        alpa.init()
        # use (1 * 4) mesh
        alpa.set_global_virtual_physical_mesh(
            alpa.get_global_cluster().get_virtual_physical_mesh([0], 4))

    def tearDown(self):
        alpa.shutdown()

    def _get_fn_manual_sharding_with(self,
                                     fn,
                                     num_microbatches,
                                     stage_option,
                                     ms_option,
                                     *args,):
        method = PipeshardParallel(
            num_micro_batches=num_microbatches,
            stage_option=stage_option,
            manual_sharding_option=ms_option,
        )
        parallelized = parallelize(fn, method=method)
        return parallelized.get_executable(*args).get_hlo_text()

    @staticmethod
    def _get_param_line(text: str):
        text = text[text.find("ENTRY"):]
        text = text[:text.find("\n")]
        return text

    @staticmethod
    def _get_root_line(text: str):
        text = text[text.find("ENTRY"):]
        text = text[text.find("ROOT"):]
        text = text[:text.find("\n")]
        return text

    @staticmethod
    def _parse_param_shapes(text: str):
        # the first one is "ENTRY %xxx ("
        params = text.split("param")[1:]
        shapes = tuple(map(lambda x: x[x.find("f32"):x.find("]") + 1], params))
        return shapes

    @staticmethod
    def _parse_root_shapes(text: str):
        tuple_shape = text[text.find("=") + 2:text.find("tuple(")]
        # the last one is ')'
        shapes = tuple_shape.split("0}")[:-1]
        shapes = tuple(map(lambda x: x[x.find("f32"):x.find("{")], shapes))
        return shapes

    def test_set_input_output(self):

        def fn(params, batch):
            x, tgt = batch

            def loss_fn(params):
                w0, b0, w1, b1, w2, b2, w3, b3 = params
                y = jax.nn.relu(x @ w0 + b0)
                z = jax.nn.relu(y @ w1 + b1)
                mark_pipeline_boundary()
                u = jax.nn.relu(z @ w2 + b2)
                v = jax.nn.softmax(u @ w3 + b3)
                return jnp.mean((v - tgt)**2)

            grads = alpa.grad(loss_fn)(params)
            new_params = tree_map(lambda p, g: p - g, params, grads)
            return new_params

        # data
        batch_size = 64
        hiddens = [6, 8, 10, 12, 14]
        params = itertools.chain(*[(jnp.ones((hiddens[i], hiddens[i + 1])),
                                    jnp.ones((hiddens[i + 1],)))
                                   for i in range(len(hiddens) - 1)])
        params = tuple(params)
        x = jnp.ones((batch_size, hiddens[0]))
        tgt = jnp.ones((batch_size, hiddens[-1]))
        batch = (x, tgt)

        # partitions
        mp_start = PartitionSpec(None, "model")
        mp_end = PartitionSpec("model", None)
        bias_partitioned = PartitionSpec("model")
        replicated = None
        dp = PartitionSpec("data", None)

        param_axis_resources = (mp_start, bias_partitioned, mp_end,
                                replicated) + (replicated, replicated,
                                               replicated, replicated)
        batch_axis_resources = (replicated, dp)
        in_axis_resources = (param_axis_resources, batch_axis_resources)

        # options
        s_option = ManualStageOption([[0], [1]], [(1, 2)] * 2, [(1, 2), (2, 1)],
                                     [{}] * 2)
        submesh_axis_names = (("dummy", "model"), ("data", "dummy"))
        ms_option = ManualShardingOption(None, submesh_axis_names,
                                         in_axis_resources)
        text = self._get_fn_manual_sharding_with(fn, 2, s_option, ms_option,
                                                 params, batch)
        print(text)
        # apply_grad_start = text.find("HloModule", 1)
        # acc_grad_text = text[:apply_grad_start]
        # apply_grad_text = text[apply_grad_start:]
        # # 1. Accumulate grad:
        # acc_grad_params = self._get_param_line(acc_grad_text)
        # acc_grad_param_shapes = self._parse_param_shapes(acc_grad_params)
        # acc_grad_root = self._get_root_line(acc_grad_text)
        # acc_grad_root_shapes = self._parse_root_shapes(acc_grad_root)

        # param_shape = ("f32[6,4]", "f32[4]", "f32[4,10]", "f32[10]")
        # # batch_size / num_microbatches / data_parallel
        # batch_shape = ("f32[16,6]", "f32[16,10]")
        # assert acc_grad_param_shapes == param_shape + batch_shape + param_shape
        # assert acc_grad_root_shapes == param_shape
        # # 2. Apply grad:
        # apply_grad_params = self._get_param_line(apply_grad_text)
        # apply_grad_param_shapes = self._parse_param_shapes(apply_grad_params)
        # apply_grad_root = self._get_root_line(apply_grad_text)
        # apply_grad_root_shapes = self._parse_root_shapes(apply_grad_root)
        # assert apply_grad_param_shapes == param_shape + param_shape
        # assert apply_grad_root_shapes == param_shape

def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipeshardManualShardingTest("test_set_input_output"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

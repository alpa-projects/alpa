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
from alpa import (AutoShardingOption, ManualShardingOption, ManualStageOption,
                  PipeshardParallel, mark_pipeline_boundary, parallelize)


class PipeshardManualShardingTest(unittest.TestCase):

    def setUp(self):
        alpa.init()
        # use (1 * 4) mesh
        alpa.set_global_virtual_physical_mesh(
            alpa.get_global_cluster().get_virtual_physical_mesh([0], 4))

    def tearDown(self):
        alpa.shutdown()

    def _get_fn_manual_sharding_with(self, fn, num_microbatches, stage_option,
                                     ms_option, *args):
        method = PipeshardParallel(
            num_micro_batches=num_microbatches,
            stage_option=stage_option,
            manual_sharding_option=ms_option,
            default_auto_sharding_option=AutoShardingOption(False))
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
        shapes = tuple(
            map(lambda x: x[x.find(": ") + 2:x.find("]") + 1], params))
        return shapes

    @staticmethod
    def _parse_root_shapes(text: str):
        tuple_shape = text[text.find("=") + 2:text.find("tuple(")]
        # the last one is ')'
        shapes = tuple_shape.split("0}")[:-1]
        shapes = tuple(map(lambda x: x[x.find("f32"):x.find("{")], shapes))
        return shapes

    @staticmethod
    def _is_superset_with_x_more(seq1, seq2, x):
        set1 = set(seq1)
        set2 = set(seq2)
        if set1.issuperset(set2) and len(set1) - len(set2) == x:
            return True
        return False

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
        s_option = ManualStageOption([[0], [1]], [(1, 2)] * 2, [(1, 2)] * 2,
                                     [{}] * 2)
        submesh_axis_names = (("dummy", "model"), ("dummy", "data"))
        ms_option = ManualShardingOption(None, submesh_axis_names,
                                         in_axis_resources)
        text = self._get_fn_manual_sharding_with(fn, 2, s_option, ms_option,
                                                 params, batch)
        l0_fwd, l1_fwd, l1_bwd, l0_bwd, l0_apl, l1_apl = text
        # layer 0
        l0_param_shape = ("f32[6,4]", "f32[4]", "f32[4,10]", "f32[10]")
        l0_batch_shape = ("f32[32,6]",)
        l0_fwd_param = self._parse_param_shapes(self._get_param_line(l0_fwd))
        assert sorted(l0_fwd_param) == sorted(l0_param_shape + l0_batch_shape)
        l0_bwd_param = self._parse_param_shapes(self._get_param_line(l0_bwd))
        l0_bwd_root = self._parse_root_shapes(self._get_root_line(l0_bwd))
        # the donated accumulated gradient are at first
        assert sorted(l0_bwd_param[:4]) == sorted(l0_param_shape)
        assert sorted(l0_bwd_root) == sorted(l0_param_shape)
        l0_apl_param = self._parse_param_shapes(self._get_param_line(l0_apl))
        l0_apl_root = self._parse_root_shapes(self._get_root_line(l0_apl))
        assert sorted(l0_apl_param) == sorted(l0_param_shape + l0_param_shape)
        assert sorted(l0_apl_root) == sorted(l0_param_shape)

        # layer 1
        l1_param_shape = ("f32[10,12]", "f32[12]", "f32[12,14]", "f32[14]")
        l1_batch_shape = ("f32[16,14]",)
        l1_fwd_param = self._parse_param_shapes(self._get_param_line(l1_fwd))
        assert self._is_superset_with_x_more(l1_fwd_param,
                                             l1_param_shape + l1_batch_shape, 1)
        l1_bwd_param = self._parse_param_shapes(self._get_param_line(l1_bwd))
        l1_bwd_root = self._parse_root_shapes(self._get_root_line(l1_bwd))
        # the donated accumulated gradient are at first
        assert sorted(l1_bwd_param[:4]) == sorted(l1_param_shape)
        assert self._is_superset_with_x_more(l1_bwd_root, l1_param_shape, 1)
        l1_apl_param = self._parse_param_shapes(self._get_param_line(l1_apl))
        l1_apl_root = self._parse_root_shapes(self._get_root_line(l1_apl))
        assert sorted(l1_apl_param) == sorted(l1_param_shape + l1_param_shape)
        assert sorted(l1_apl_root) == sorted(l1_param_shape)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipeshardManualShardingTest("test_set_input_output"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

"""Test the numerical correctness of shard parallel."""
import unittest

import jax

from alpa.pipeline_parallel.stage_construction import ManualStageOption
from alpa.testing import PipelineBasicTest


class AccumulateGradTest(PipelineBasicTest):

    def test_mlp(self):
        self.run_mlp()

    def test_2_layer_bert(self):
        self.run_n_layer_bert(num_layers=2)

    @unittest.skipIf(jax.device_count('gpu') < 8, "no enough device")
    def test_8_layer_bert(self):
        self.run_n_layer_bert(num_layers=8)

    @unittest.skipIf(jax.device_count('gpu') < 8, "no enough device")
    def test_8_layer_bert_manual_stage_assignment(self):
        stage_option = ManualStageOption(forward_stage_layer_ids=[[0, 1, 2, 3],
                                                                  [4, 5, 6, 7]],
                                         submesh_physical_shapes=[(1, 4),
                                                                  (1, 4)],
                                         submesh_logical_shapes=None,
                                         submesh_autosharding_option_dicts=None)
        self.run_n_layer_bert(num_layers=8, stage_option=stage_option)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AccumulateGradTest('test_mlp'))
    suite.addTest(AccumulateGradTest('test_2_layer_bert'))
    suite.addTest(AccumulateGradTest('test_8_layer_bert'))
    suite.addTest(
        AccumulateGradTest('test_8_layer_bert_manual_stage_assignment'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

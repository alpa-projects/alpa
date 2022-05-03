import unittest
import jax

from alpa.testing import PipelineBasicTest


class StageConstructionTest(PipelineBasicTest):

    def test_mlp(self):
        self.run_mlp(return_value=True)

    def test_2_layer_bert(self):
        self.run_n_layer_bert(n_layers=2, return_value=True)

    @unittest.skipIf(jax.device_count('gpu') < 8, "no enough device")
    def test_8_layer_bert(self):
        self.run_n_layer_bert(n_layers=8, return_value=True)

    @unittest.skipIf(jax.device_count('gpu') < 8, "no enough device")
    def test_8_layer_bert_manual_stage_assignment(self):
        self.run_n_layer_bert(n_layers=8,
                              pipeline_stage_mode="manual_stage",
                              forward_stage_layer_ids=[[0, 1, 2, 3],
                                                       [4, 5, 6, 7]],
                              submesh_shapes=[(1, 4), (1, 4)])


def suite():
    suite = unittest.TestSuite()
    suite.addTest(StageConstructionTest('test_mlp'))
    suite.addTest(StageConstructionTest('test_2_layer_bert'))
    suite.addTest(StageConstructionTest('test_8_layer_bert'))
    suite.addTest(
        StageConstructionTest('test_8_layer_bert_manual_stage_assignment'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

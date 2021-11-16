import unittest

import jax
from parax.testing import PipelineBasicTest


class PipelineRematTest(PipelineBasicTest):

    # FIXME
    @unittest.skip("Flaky because of layer slicing")
    def test_mlp_remat(self):
        self.run_mlp(test_remat=True)

    def test_2_layer_bert_remat(self):
        self.run_n_layer_bert(n_layers=2, test_remat=True)

    def test_2_layer_bert_auto_layer_slicing_remat(self):
        self.run_n_layer_bert(n_layers=2,
                              manual_pipeline_layer=False,
                              test_remat=True)

    @unittest.skipIf(jax.device_count('gpu') < 8, "no enough device")
    def test_8_layer_bert_auto_layer_slicing_remat(self):
        self.run_n_layer_bert(n_layers=8,
                              manual_pipeline_layer=False,
                              test_remat=True)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineRematTest('test_mlp_remat'))
    suite.addTest(PipelineRematTest('test_2_layer_bert_remat'))
    suite.addTest(
        PipelineRematTest('test_2_layer_bert_auto_layer_slicing_remat'))
    suite.addTest(
        PipelineRematTest('test_8_layer_bert_auto_layer_slicing_remat'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

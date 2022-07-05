import unittest

import jax
from alpa.testing import PipelineBasicTest


class LayerConstructionTest(PipelineBasicTest):

    def test_mlp_layer_construction(self):
        self.run_mlp(manual_pipeline_layer=False)

    def test_2_layer_bert_layer_construction(self):
        self.run_n_layer_bert(num_layers=2, manual_pipeline_layer=False)

    @unittest.skipIf(jax.device_count('gpu') < 8, "no enough device")
    def test_8_layer_bert_layer_construction(self):
        self.run_n_layer_bert(num_layers=8, manual_pipeline_layer=False)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(LayerConstructionTest('test_mlp_layer_construction'))
    suite.addTest(LayerConstructionTest('test_2_layer_bert_layer_construction'))
    suite.addTest(LayerConstructionTest('test_8_layer_bert_layer_construction'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

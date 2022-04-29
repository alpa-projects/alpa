import unittest
import time

import jax

from alpa.testing import PipelineBasicTest


class AccumulateGradTest(PipelineBasicTest):

    def test_mlp(self):
        self.run_mlp(pipeline_stage_mode='auto_gpipe', num_micro_batches='auto')

    def test_2_layer_bert(self):
        self.run_n_layer_bert(n_layers=2,
                              pipeline_stage_mode='auto_gpipe',
                              num_micro_batches='auto')

    @unittest.skipIf(jax.device_count('gpu') < 8, "no enough device")
    def test_8_layer_bert(self):
        self.run_n_layer_bert(n_layers=8,
                              pipeline_stage_mode='auto_gpipe',
                              num_micro_batches='auto')


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AccumulateGradTest('test_mlp'))
    # suite.addTest(AccumulateGradTest('test_2_layer_bert'))
    # suite.addTest(AccumulateGradTest('test_8_layer_bert'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

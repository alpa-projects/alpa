import unittest

from parax.testing import PipelineBasicTest


class AutoStageClusteringOOMTest(PipelineBasicTest):

    def test_GPT3_3B(self):
        self.run_n_layer_bert(n_layers=32,
                              hidden_size=2560,
                              batch_size=16,
                              seq_len=1024,
                              num_heads=32,
                              pipeline_stage_mode="auto_gpipe",
                              cache_compute_cost=None)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoStageClusteringOOMTest('test_GPT3_27B'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

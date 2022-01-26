import unittest

from alpa.testing import PipelineBasicTest


class StageConstructionTest(PipelineBasicTest):

    def test_mlp_stage_construction(self):
        self.run_mlp(pipeline_stage_mode="auto_gpipe")

    def test_mlp_layer_and_stage(self):
        self.run_mlp(manual_pipeline_layer=False,
                     pipeline_stage_mode="auto_gpipe")

    def test_2_layer_bert_stage_construction(self):
        self.run_n_layer_bert(n_layers=2, pipeline_stage_mode="auto_gpipe")

    def test_2_layer_bert_layer_and_stage(self):
        self.run_n_layer_bert(n_layers=2,
                              manual_pipeline_layer=False,
                              pipeline_stage_mode="auto_gpipe")

    def test_8_layer_bert_stage_construction(self):
        self.run_n_layer_bert(n_layers=8,
                              pipeline_stage_mode="auto_gpipe",
                              cache_compute_cost=None)

    def test_8_layer_bert_layer_and_stage(self):
        self.run_n_layer_bert(n_layers=8,
                              manual_pipeline_layer=False,
                              pipeline_stage_mode="auto_gpipe",
                              cache_compute_cost=None)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(StageConstructionTest('test_mlp_stage_construction'))
    suite.addTest(StageConstructionTest('test_mlp_layer_and_stage'))
    suite.addTest(StageConstructionTest('test_2_layer_bert_stage_construction'))
    suite.addTest(StageConstructionTest('test_2_layer_bert_layer_and_stage'))
    suite.addTest(StageConstructionTest('test_8_layer_bert_stage_construction'))
    suite.addTest(StageConstructionTest('test_8_layer_bert_layer_and_stage'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

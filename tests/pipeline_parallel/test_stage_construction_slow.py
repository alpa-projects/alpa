import unittest

from alpa.pipeline_parallel.stage_construction import AutoStageOption
from alpa.testing import PipelineBasicTest


def auto_stage():
    return AutoStageOption("small_power_of_two", "same_as_physical",
                           float("inf"), False, None, None)


class StageConstructionSlowTest(PipelineBasicTest):

    def test_mlp_stage_construction(self):
        self.run_mlp(stage_option=auto_stage())

    def test_mlp_layer_and_stage(self):
        self.run_mlp(manual_pipeline_layer=False, stage_option=auto_stage())

    def test_2_layer_bert_stage_construction(self):
        self.run_n_layer_bert(num_layers=2, stage_option=auto_stage())

    def test_2_layer_bert_layer_and_stage(self):
        self.run_n_layer_bert(num_layers=2,
                              manual_pipeline_layer=False,
                              stage_option=auto_stage())

    def test_8_layer_bert_stage_construction(self):
        self.run_n_layer_bert(num_layers=8, stage_option=auto_stage())

    def test_8_layer_bert_layer_and_stage(self):
        self.run_n_layer_bert(num_layers=8,
                              manual_pipeline_layer=False,
                              stage_option=auto_stage())


def suite():
    suite = unittest.TestSuite()
    suite.addTest(StageConstructionSlowTest('test_mlp_stage_construction'))
    suite.addTest(StageConstructionSlowTest('test_mlp_layer_and_stage'))
    suite.addTest(
        StageConstructionSlowTest('test_2_layer_bert_stage_construction'))
    suite.addTest(
        StageConstructionSlowTest('test_2_layer_bert_layer_and_stage'))
    suite.addTest(
        StageConstructionSlowTest('test_8_layer_bert_stage_construction'))
    suite.addTest(
        StageConstructionSlowTest('test_8_layer_bert_layer_and_stage'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

import unittest

from alpa.pipeline_parallel.stage_construction import AutoStageOption
from alpa.testing import PipelineBasicTest


def auto_stage():
    return AutoStageOption("small_power_of_two", "same_as_physical",
                           float("inf"), False, None, None)


class StageConstructionTest(PipelineBasicTest):

    def test_mlp_stage_construction(self):
        self.run_mlp(stage_option=auto_stage())

    def test_mlp_layer_and_stage(self):
        self.run_mlp(manual_pipeline_layer=False, stage_option=auto_stage())


def suite():
    suite = unittest.TestSuite()
    suite.addTest(StageConstructionTest('test_mlp_stage_construction'))
    suite.addTest(StageConstructionTest('test_mlp_layer_and_stage'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

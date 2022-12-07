import unittest
from alpa import init, PipeshardParallel, AutoStageOption
from tests.pipeline_parallel.test_inference_only import PipelineInferenceTest


class PipelineInferenceAutoTest(PipelineInferenceTest):

    def setUp(self):
        init(cluster="ray", num_nodes=1, num_devices_per_node=4)

    def test_mlp(self):
        stage_option = AutoStageOption(
            submesh_physical_shape_space="manual",
            manually_specified_submeshes=((1, 2),),
            submesh_logical_shape_space="model_parallel_only")
        method = PipeshardParallel(num_micro_batches=1,
                                   pipeline_schedule="inference",
                                   layer_option="manual",
                                   stage_option=stage_option)
        self.run_mlp_inference(True, method)

    def test_bert(self):
        stage_option = AutoStageOption(
            submesh_physical_shape_space="manual",
            manually_specified_submeshes=((1, 2),),
            submesh_logical_shape_space="model_parallel_only")
        method = PipeshardParallel(num_micro_batches=1,
                                   pipeline_schedule="inference",
                                   layer_option="manual",
                                   stage_option=stage_option)
        self.run_bert_layer_collection_inference(True, method)

    def test_mlp_1d(self):
        stage_option = AutoStageOption(
            submesh_physical_shape_space="manual",
            manually_specified_submeshes=((1, 2),),
            submesh_logical_shape_space="model_parallel_only",
            layer_profile_mode="individual")
        method = PipeshardParallel(num_micro_batches=1,
                                   pipeline_schedule="inference",
                                   layer_option="manual",
                                   stage_option=stage_option)
        self.run_mlp_inference(True, method)

    def test_bert_1d(self):
        stage_option = AutoStageOption(
            submesh_physical_shape_space="manual",
            manually_specified_submeshes=((1, 2),),
            submesh_logical_shape_space="model_parallel_only",
            layer_profile_mode="individual")
        method = PipeshardParallel(num_micro_batches=1,
                                   pipeline_schedule="inference",
                                   layer_option="manual",
                                   stage_option=stage_option)
        self.run_bert_layer_collection_inference(True, method)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineInferenceAutoTest("test_mlp"))
    suite.addTest(PipelineInferenceAutoTest("test_bert"))
    suite.addTest(PipelineInferenceAutoTest("test_mlp_1d"))
    suite.addTest(PipelineInferenceAutoTest("test_bert_1d"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

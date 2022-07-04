import unittest

from alpa.device_mesh import (get_global_cluster,
                              set_global_virtual_physical_mesh)
from alpa.pipeline_parallel.stage_construction import ManualStageOption
from alpa.testing import PipelineBasicTest


class ScatterGatherTest(PipelineBasicTest):

    def test_2_layer_bert(self):
        virtual_mesh = get_global_cluster().get_virtual_physical_mesh([0], 4)
        set_global_virtual_physical_mesh(virtual_mesh)

        stage_option = ManualStageOption(
            forward_stage_layer_ids=[[0], [1]],
            submesh_physical_shapes=[(1, 2), (1, 2)],
            submesh_logical_shapes=[(1, 2), (2, 1)],
            submesh_autosharding_option_dicts=[
                dict(force_batch_dim_to_mesh_dim=0), {}
            ])

        self.run_n_layer_bert(num_layers=2,
                              batch_size=4,
                              seq_len=4,
                              hidden_size=4,
                              num_heads=1,
                              stage_option=stage_option)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(ScatterGatherTest('test_2_layer_bert'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

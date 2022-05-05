import unittest

from alpa import DeviceCluster
from alpa.testing import PipelineBasicTest


class ScatterGatherTest(PipelineBasicTest):

    def test_2_layer_bert(self):
        virtual_mesh = DeviceCluster().get_virtual_physical_mesh([0], 4)

        self.run_n_layer_bert(n_layers=2,
                              batch_size=4,
                              seq_len=4,
                              hidden_size=4,
                              num_heads=1,
                              pipeline_stage_mode="manual_stage",
                              forward_stage_layer_ids=[[0], [1]],
                              overwrite_global_config_dict=dict(
                                  sub_physical_mesh_shapes=[(1, 2)] * 2,
                                  sub_logical_mesh_shapes=[(1, 2), (2, 1)],
                                  submesh_autosharding_option_dicts=[
                                      dict(force_batch_dim_to_mesh_dim=0), {}
                                  ],
                                  use_scatter_gather=True),
                              virtual_mesh=virtual_mesh)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(ScatterGatherTest('test_2_layer_bert'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

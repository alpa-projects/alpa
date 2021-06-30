import unittest
import numpy as np

from parax.cross_mesh_resharding import unflatten_tile_index

class CrossMeshReshardingUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        print("test")

    def test_unflattend_tile_index(self):
        index = 8
        shape = [5, 2]
        unflattend_index = unflatten_tile_index(index, shape)
        assert unflattend_index == [4, 0]



def suite():
    suite = unittest.TestSuite()
    suite.addTest(CrossMeshReshardingUnitTest('test_unflattend_tile_index'))


    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
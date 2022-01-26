"""Test OrderedSet."""

import os
import unittest

from alpa.util import OrderedSet


class OrderedSetTest(unittest.TestCase):
    """Test OrderedSet."""

    def test_init(self):
        """Test OrderedSet.__init__."""
        oset = OrderedSet()
        self.assertEqual(len(oset), 0)

        oset = OrderedSet([1, 2, 3])
        self.assertEqual(len(oset), 3)

    def test_add(self):
        """Test OrderedSet.add."""
        oset = OrderedSet()
        oset.add(1)
        self.assertEqual(len(oset), 1)

        oset.add(2)
        self.assertEqual(len(oset), 2)

    def test_update(self):
        """Test OrderedSet.update."""
        oset = OrderedSet([1, 2, 3])
        oset.update([4, 5])
        self.assertEqual(len(oset), 5)
        self.assertEqual(oset, OrderedSet([1, 2, 3, 4, 5]))

    def test_union(self):
        """Test OrderedSet.union."""
        oset = OrderedSet([1, 2, 3])
        self.assertEqual(oset.union([4, 5]), OrderedSet([1, 2, 3, 4, 5]))

    def test_intersection_update(self):
        """Test OrderedSet.intersection_update."""
        oset = OrderedSet([1, 2, 3])
        oset.intersection_update([2, 3, 4])
        self.assertEqual(len(oset), 2)
        self.assertEqual(oset, OrderedSet([2, 3]))

        oset = OrderedSet([1, 2, 3])
        oset.intersection_update([2, 3, 4])
        self.assertEqual(len(oset), 2)
        self.assertEqual(oset, OrderedSet([2, 3]))

    def test_intersection(self):
        """Test OrderedSet.intersection."""
        oset = OrderedSet([1, 2, 3])
        result = oset.intersection([2, 3, 4])
        self.assertEqual(len(result), 2)
        self.assertEqual(result, OrderedSet([2, 3]))

    def test_remove(self):
        """Test OrderedSet.remove."""
        oset = OrderedSet([1, 2, 3])
        oset.remove(2)
        self.assertEqual(len(oset), 2)
        self.assertEqual(oset, OrderedSet([1, 3]))

    def test_discard(self):
        """Test OrderedSet.discard."""
        oset = OrderedSet([1, 2, 3])
        oset.discard(2)
        self.assertEqual(len(oset), 2)
        self.assertEqual(oset, OrderedSet([1, 3]))

        oset.discard(4)
        self.assertEqual(len(oset), 2)
        self.assertEqual(oset, OrderedSet([1, 3]))

    def test_clear(self):
        """Test OrderedSet.clear."""
        oset = OrderedSet([1, 2, 3])
        oset.clear()
        self.assertEqual(len(oset), 0)

    def test_difference(self):
        """Test OrderedSet.difference."""
        oset = OrderedSet([1, 2, 3])
        result = oset.difference([2, 3, 4])
        self.assertEqual(len(result), 1)
        self.assertEqual(result, OrderedSet([1]))

    def test_difference_update(self):
        """Test OrderedSet.difference_update."""
        oset = OrderedSet([1, 2, 3])
        oset.difference_update([2, 3, 4])
        self.assertEqual(len(oset), 1)
        self.assertEqual(oset, OrderedSet([1]))

    def test_symmetric_difference(self):
        """Test OrderedSet.symmetric_difference."""
        oset = OrderedSet([1, 2, 3])
        result = oset.symmetric_difference([2, 3, 4])
        self.assertEqual(len(result), 2)
        self.assertEqual(result, OrderedSet([1, 4]))

    def test_repr(self):
        """Test OrderedSet.__repr__."""
        oset = OrderedSet([1, 2, 3])
        self.assertEqual(repr(oset), 'OrderedSet([1, 2, 3])')


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(OrderedSetTest))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

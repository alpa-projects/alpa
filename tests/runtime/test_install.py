import unittest

from alpa.test_install import suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

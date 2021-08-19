"""Run all test cases.
Run each file in a separate process to avoid GPU memory conflicts.
"""

import glob
import multiprocessing
import unittest


def run_unittest_files(files):
    """Run unit test files one by one in separates processes."""
    for filename in files:
        if not filename.startswith("test"):
            continue

        def func():
            ret = unittest.main(module=None, argv=["", "-vb"] + [filename])

        p = multiprocessing.Process(target=func)
        p.start()
        p.join()

        if p.exitcode != 0:
            return


if __name__ == "__main__":
    files = glob.glob("*.py")
    files.sort()
    run_unittest_files(files)

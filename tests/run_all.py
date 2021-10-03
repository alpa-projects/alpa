"""Run all test cases.
Run each file in a separate process to avoid GPU memory conflicts.
"""

import argparse
import glob
import multiprocessing
import time
from typing import Sequence
import unittest


def run_unittest_files(files, args):
    """Run unit test files one by one in separates processes."""
    for filename in files:
        if not filename.startswith("test"):
            continue
        if args.only_pipeline and not filename.count('pipeline'):
            continue

        def func():
            ret = unittest.main(module=None, argv=["", "-vb"] + [filename])

        p = multiprocessing.Process(target=func)
        p.start()
        p.join()

        if p.exitcode != 0:
            return


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--only-pipeline',
                            action='store_true',
                            help='only run tests with name pipeline')
    args = arg_parser.parse_args()

    files = glob.glob("*.py")
    files.sort()

    tic = time.time()
    run_unittest_files(files, args)
    print(f"Run all tests in {time.time() - tic:.2f}s")

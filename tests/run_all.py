"""Run all test cases.
Run each file in a separate process to avoid GPU memory conflicts.

Usages:
python3 run_all.py
python3 run_all.py --filter pipeline
python3 run_all.py --filter auto_sharding
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
        if args.filter is not None and args.filter not in filename:
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
    arg_parser.add_argument("--filter", type=str, default=None,
                            help="Run test cases whose names contain the filter string")
    args = arg_parser.parse_args()

    files = glob.glob("*.py")
    files.sort()

    tic = time.time()
    run_unittest_files(files, args)
    print(f"Run all tests in {time.time() - tic:.2f}s")

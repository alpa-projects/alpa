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
import os
import numpy as np
import time
from typing import Sequence
import unittest

from alpa.util import run_with_timeout

slow_testcases = set([
    "test_pipeline_stage_construction.py",
])


def run_unittest_files(files, args):
    """Run unit test files one by one in separates processes."""
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
        args.xla_client_mem_fraction)

    for filename in files:
        if not filename.startswith("test"):
            continue
        if args.filter is not None and args.filter not in filename:
            continue
        if not args.enable_slow_tests and filename in slow_testcases:
            continue

        def func():
            ret = unittest.main(module=None, argv=["", "-vb"] + [filename])

        p = multiprocessing.Process(target=func)

        def run_one_file():
            p.start()
            p.join()

        try:
            run_with_timeout(run_one_file, timeout=args.time_limit_per_file)
            if p.exitcode != 0:
                return False
        except TimeoutError:
            p.terminate()
            time.sleep(5)
            print(f"\nTimeout after {args.time_limit_per_file} seconds "
                  f"when running {filename}")
            return False

    return True


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Run test cases whose names contain the filter string")
    arg_parser.add_argument(
        "--enable-slow-tests",
        action="store_true",
        help="Run test cases including profiling, which takes a long time")
    arg_parser.add_argument(
        "--xla-client-mem-fraction",
        type=float,
        default=0.2,
        help="The fraction of GPU memory used to run unit tests")
    arg_parser.add_argument(
        "--time-limit-per-file",
        type=int,
        default=1000,
        help="The time limit for running one file in seconds.")
    arg_parser.add_argument("--order",
                            type=str,
                            default="sorted",
                            choices=["sorted", "random", "reverse_sorted"])
    args = arg_parser.parse_args()

    files = glob.glob("*.py")
    if args.order == "sorted":
        files.sort()
    elif args.order == "random":
        files = [files[i] for i in np.random.permutation(len(files))]
    elif args.order == "reverse_sorted":
        files.sort()
        files = reversed(files)

    tic = time.time()
    success = run_unittest_files(files, args)

    if success:
        print(f"Success. Time elapsed: {time.time() - tic:.2f}s")
    else:
        print(f"Fail. Time elapsed: {time.time() - tic:.2f}s")

    exit(0 if success else -1)

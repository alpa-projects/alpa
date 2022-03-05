from concurrent.futures import thread
from functools import reduce
import sys
import subprocess
import multiprocessing
import threading
import tqdm
import os


class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        if sys.version_info >= (3, 0):
            threading.Thread.join(self, *args)
        else:
            threading.Thread.join(self)

        return self._return


def parallize_v2(f, args, desc='threading'):
    threads = multiprocessing.cpu_count()
    print(f"number of threads: {threads}")
    # threads = 1
    return parallize(f, args, threads=threads, desc=desc)


def parallize(f, args, threads=None, desc='threading'):
    """
        Args:
            - f: function
            - args: list or list(tuple), list when threads not None
    """
    def parse_arg(arg):
        if type(arg) == list or type(arg) == set or type(arg) == tuple:
            return tuple(arg)
        elif type(arg) == dict:
            return arg
        else:
            return (arg, )

    if threads is not None:

        results = []
        for i in tqdm.trange(0, len(args), threads, desc=desc):
            func_args = [parse_arg(arg) for arg in args[i:i + threads]]
            if len(func_args) == 0:
                continue

            if type(func_args[0]) == dict:
                active_threads = [ThreadWithReturnValue(
                    target=f, kwargs=arg) for arg in func_args]
            else:
                active_threads = [ThreadWithReturnValue(
                    target=f, args=arg) for arg in func_args]
            [thread.start() for thread in active_threads]
            results += [thread.join() for thread in active_threads]
        return results
    else:
        if len(args) == 0:
            return []

        if type(args[0]) == dict:
            active_threads = [ThreadWithReturnValue(
                target=f, kwargs=arg) for arg in args]
        else:
            args = [parse_arg(arg) for arg in args]
            active_threads = [ThreadWithReturnValue(
                target=f, args=arg) for arg in args]

        [thread.start() for thread in active_threads]
        return [thread.join() for thread in active_threads]


def parallize_v3(f, args, n_processes=None, desc='parallize_v3'):
    if n_processes == None:
        n_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(n_processes) as pool:
        results = [r for r in tqdm.tqdm(pool.imap(f, args), desc=desc, total=len(args))]

    return results

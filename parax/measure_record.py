"""Measurement records that serialize the tasks, strategies, and measurement results to files."""
from collections import namedtuple
import json
import os
from typing import Tuple

import numpy as np

from parax.util import to_int_tuple

RECORD_VERSION = "v0.1"


class SearchTask:
    """
    The input specification of an auto-parallelization search task.

    The input coantains a computation specification and a device cluster specification.
    """

    def __init__(self, compute_key, device_key):
        self.compute_key = compute_key
        self.device_key = device_key

    def get_task_key(self) -> str:
        """Return a unique string as the query key of this task."""
        return f"({self.compute_key}, {self.device_key})"

    def to_jsonable(self):
        return (self.compute_key, self.device_key)

    @staticmethod
    def from_jsonable(value):
        compute_key, device_key = value
        return SearchTask(compute_key, device_key)


class StrategyConfig:
    """A configuration that specifies all details of a parallelization strategy."""

    def __init__(self, build_random_seed: int, logical_mesh_shape: Tuple[int],
                 all_reduce_threshold: int,
                 auto_sharding_solution_vector: np.ndarray):
        self.build_random_seed = build_random_seed
        self.logical_mesh_shape = logical_mesh_shape
        self.all_reduce_threshold = all_reduce_threshold
        self.auto_sharding_solution_vector = auto_sharding_solution_vector

    def to_jsonable(self):
        return (self.build_random_seed, tuple(self.logical_mesh_shape),
                self.all_reduce_threshold,
                to_int_tuple(self.auto_sharding_solution_vector))

    @staticmethod
    def from_jsonable(value):
        (build_random_seed, logical_mesh_shape, all_reduce_threshold,
         auto_sharding_solution_vector) = value
        return StrategyConfig(build_random_seed, logical_mesh_shape,
                              all_reduce_threshold,
                              np.array(auto_sharding_solution_vector))


class MeasureInput(namedtuple("MeasureInput", ["task", "config"])):
    """
    Stores all the inputs of a measurement.

    Args:
      task (SearchTask): The search task.
      config (StrategyConfig): The parallelization strategy.
    """


class MeasureResult(
        namedtuple("MeasureResult",
                   ["time_costs", "estimated_cost", "error_no", "timestamp"])):
    """
    Stores all the results of a measurement.

    Args:
      time_costs (List[float]): The measured execution time.
      estimated_cost: (float): The estimated cost by the cost model.
      error_no (int): The error code.
      timestamp (int): The time stamp of measurement.
    """


def save_to_file(inputs, results, filename, protocol="json"):
    """
    Save measurement records to a file.

    Args:
      inputs (List[MeasureInput]):
      results (List[MeasureResult]):
      filename (str):
      protocol (str):
    """
    assert protocol == "json"

    with open(filename, "a") as fout:
        for inp, res in zip(inputs, results):
            obj = (inp.task.to_jsonable(), inp.config.to_jsonable(),
                   res.time_costs, res.estimated_cost, res.error_no,
                   res.timestamp, RECORD_VERSION)
            fout.write(json.dumps(obj) + "\n")


def load_from_file(filename, protocol="json"):
    """
    Load measurement records from a file.

    Args:
      filename (str):
      protocol (str):

    Yields:
      inp (MeasureInput):
      res (MeasureResult):
    """
    assert protocol == "json"

    if not os.path.exists(filename):
        return

    for line in open(filename, "r"):
        obj = json.loads(line)
        (task_jsonable, config_jsonable, time_costs, estimated_cost, error_no,
         timestamp, _) = obj

        inp = MeasureInput(SearchTask.from_jsonable(task_jsonable),
                           StrategyConfig.from_jsonable(config_jsonable))
        res = MeasureResult(time_costs, estimated_cost, error_no, timestamp)
        yield inp, res


def load_best_record(search_task, filename):
    """
    Load the the best record for a search task. Best means the lowest time cost.

    Args:
      search_task (SearchTask):
      filename (str):

    Returns:
      inp (MeasureInput):
      res (MeasureResult):
    """
    task_key = search_task.get_task_key()

    best_inp, best_res = None, None
    best_cost = float("inf")

    for inp, res in load_from_file(filename):
        if inp.task.get_task_key() == task_key:
            cost = np.mean(res.time_costs)
            if cost < best_cost:
                best_inp, best_res = inp, res
                best_cost = cost

    return best_inp, best_res

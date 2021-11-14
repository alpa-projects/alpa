"""Generate pipeline schedules."""

from typing import List

import numpy as np

from parax.pipeline_parallel.computation import PipelineComputation
from parax.util import cached_property, OrderedSet


def gen_dependency_with_stages(compute_stages: List[PipelineComputation],
                               n_apply_grad_stages=0,
                               apply_grad_deps=()):
    """Generate the dependency matrix for a list of pipeline stages."""
    n_stages = len(compute_stages) + n_apply_grad_stages
    d = np.zeros([n_stages, n_stages], dtype=np.int)
    var_stage_id = {}
    for i, stage in enumerate(compute_stages):
        for var in stage.invars:
            if var in var_stage_id:
                d[i, var_stage_id[var]] = 1
            else:
                # Assume the var is from global_invars
                pass
        for var in stage.outvars:
            var_stage_id[var] = i

    # TODO(yonghao): this can be inferred as well.
    for apply_stage_id, compute_stage_id in apply_grad_deps:
        d[apply_stage_id][compute_stage_id] = 1
    return d


class GpipeSchedule:
    """
    Construct a Gpipe-like schedule.

    Args:
        dependency (np.array): dependency adjacency matrix.
        sliced_mesh (List[VirtualMesh]): a list of pre-sliced virtual meshes
            to assign workers on.
        apply_grad_placement (Dict[int, int]): A map from apply grad's stage idx
            to the worker it is assigned.
        num_batch (int): number of microbatches.
    """

    def __init__(self,
                 *,
                 dependency,
                 sliced_meshes,
                 apply_grad_placement,
                 num_batch=1):
        self.dependency = dependency
        self.meshes = sliced_meshes
        self.apply_grad_placement = apply_grad_placement
        self.num_batch = num_batch
        self.num_stage = dependency.shape[0]
        self.num_worker = len(sliced_meshes)
        self.num_mesh = len(sliced_meshes)
        self._schedules = self._generate_schedule()

    def _generate_schedule(self):
        """
        Generate a Gpipe-like schedule.

        Note that here we always assume num_pipeline_workers = num_stage / 2.

        The schedule will look like below:
        i: index of micro-batch
        j: index of partition/device
        k: clock number

        k (i,j) (i,j) (i,j)
        - ----- ----- -----
        0 (0,0)
        1 (1,0) (0,1)
        2 (2,0) (1,1) (0,2)
        3       (2,1) (1,2)
        4             (2,2)
        5 reverse...
        """
        m = self.num_batch
        n = self.num_worker
        num_clock = m + n - 1
        schedules = []
        for k in range(num_clock):
            scheds = [None] * n
            for d in range(max(1 + k - m, 0), min(1 + k, n)):
                scheds[d] = (k - d, d)
            schedules.append(scheds)

        def reverse(scheds):
            rev = []
            for task in scheds:
                if not task:
                    rev.append(None)
                else:
                    rev.append((task[0], 2 * n - 1 - task[1]))
            return rev

        # backward schedules
        for k in range(num_clock):
            mapped_scheds = schedules[num_clock - k - 1]
            schedules.append(reverse(mapped_scheds))

        # apply_grad schedules
        scheds = [None] * n
        for stage_idx, worker in self.apply_grad_placement.items():
            scheds[worker] = (0, stage_idx)
        schedules.append(scheds)
        return schedules

    def pprint_schedule(self):
        """Pretty print the schedule."""
        printout = "\n"
        device_str = " ".join([
            "{:<8}".format("d" + str(d))
            for d in range(self.num_pipeline_worker)
        ])
        printout = printout + "Clock {:<2}: {} \n".format("k", device_str)
        for clock, scheds in enumerate(self.schedules):
            sched_str = " ".join(
                ["{:<8}".format(str(sched)) for sched in scheds])
            printout = printout + "Clock {:<2}: {} \n".format(clock, sched_str)
        return printout

    @cached_property
    def stage_worker_mapping(self):
        """Generate a stage-worker mapping according to the schedule."""
        placements = dict()
        for tasks in self._schedules:
            for worker_idx, task in enumerate(tasks):
                if task:
                    _, stage_idx = task
                    if stage_idx not in placements:
                        placements[stage_idx] = OrderedSet()
                    if worker_idx not in placements[stage_idx]:
                        placements[stage_idx].add(worker_idx)
        return placements

    @cached_property
    def worker_stage_mapping(self):
        """Generate a worker-stage mapping according to the schedule."""
        ownership = dict()
        for tasks in self._schedules:
            for worker_idx, task in enumerate(tasks):
                if task:
                    _, stage_idx = task
                    if worker_idx not in ownership:
                        ownership[worker_idx] = OrderedSet()
                    if stage_idx not in ownership[worker_idx]:
                        ownership[worker_idx].add(stage_idx)
        return ownership

    def stage_placement(self, stage_idx):
        """Query the placement of a stage given its stage index."""
        return self.stage_worker_mapping[stage_idx]

    def worker_placement(self, worker_idx):
        """Query the responsible stages of a worker given a worker index."""
        return self.worker_stage_mapping[worker_idx]

    @property
    def schedules(self):
        """Return the schedules as a matrix."""
        return self._schedules

    @property
    def num_clock(self):
        """Return the number of clocks in the schedule."""
        return len(self._schedules)

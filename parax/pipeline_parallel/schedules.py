"""Generate pipeline schedules."""

from typing import List, Tuple

import numpy as np

from parax.pipeline_parallel.computation import PipelineComputation
from parax.util import cached_property


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


class PipelineSchedule:
    """
    A pipeline schedule used by the distributed runtime.

    The core interface of this schedule is .schedule object

    Args:
        dependency (np.array): dependency adjacency matrix.
        sliced_mesh (List[VirtualMesh]): a list of pre-sliced virtual meshes
            to assign stages on.
        apply_grad_placement (Dict[int, int]): A map from apply grad's stage idx
            to the worker it is assigned.
        num_batch (int): number of microbatches.
    """

    def __init__(self,
                 *,
                 dependency,
                 meshes,
                 apply_grad_placement,
                 num_batch=1):
        self.dependency = dependency
        self.meshes = meshes
        self.apply_grad_placement = apply_grad_placement
        self.num_batch = num_batch

        self._schedules : List[List[Tuple]] = self._generate_schedule()

    def _generate_schedule(self):
        raise NotImplementedError()

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

    @property
    def schedules(self):
        """Return the schedules."""
        return self._schedules

    @property
    def num_stage(self):
        """Return the number of stage, including apply_grad stages."""
        return self.dependency.shape[0]

    @property
    def num_mesh(self):
        return len(self.meshes)

    @property
    def num_clock(self):
        """Return the number of clocks in the schedule."""
        return len(self._schedules)

    @cached_property
    def stage_mesh_mapping(self):
        """Generate a stage-worker mapping according to the schedule."""
        placements = dict()
        for tasks in self._schedules:
            for mesh_idx, task in enumerate(tasks):
                if task:
                    _, stage_idx = task
                    if stage_idx not in placements:
                        placements[stage_idx] = set()
                    if mesh_idx not in placements[stage_idx]:
                        placements[stage_idx].add(mesh_idx)
        return placements

    @cached_property
    def mesh_stage_mapping(self):
        """Generate a worker-stage mapping according to the schedule."""
        ownership = dict()
        for tasks in self._schedules:
            for mesh_idx, task in enumerate(tasks):
                if task:
                    _, stage_idx = task
                    if mesh_idx not in ownership:
                        ownership[mesh_idx] = set()
                    if stage_idx not in ownership[mesh_idx]:
                        ownership[mesh_idx].add(stage_idx)
        return ownership

    def stage_placement(self, stage_idx):
        """Query the placement of a stage given its stage index."""
        return self.stage_mesh_mapping[stage_idx]

    def mesh_placement(self, worker_idx):
        """Query the responsible stages of a worker given a worker index."""
        return self.mesh_stage_mapping[worker_idx]


class GpipeSchedule(PipelineSchedule):
    """Construct a Gpipe-like schedule."""

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
        n = self.num_mesh
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


class PipeDreamFlush(PipelineSchedule):
    """
    Generate a PipeDream-Flush schedule (a.k.a. 1F1B).

    It has similar latency to GPipe but is more memory-efficient.
    """
    def _generate_schedule(self):
        m = self.num_batch
        n = self.num_mesh

        # equal to gpipe
        num_clock = (m + n - 1) * 2
        schedules = [[None] * n for k in range(num_clock)]

        num_warmup_microbatches = [min(m - i - 1, m) for i in range(n)]
        num_microbatches_remaining = [m - i for i in num_warmup_microbatches]


        # warm-up clocks
        M = max(num_warmup_microbatches)
        for k in range(M):
            for d in range(max(1 + k - M, 0), min(1 + k, n)):
                schedules[k][d] = (k - d, d)

        next_fwd_mb_idx = num_warmup_microbatches
        next_bwd_mb_idx = [0 for _ in range(n)]
        next_available_clock = [M for _ in range(n)]


        finished_bwd_batch_indices = np.zeros(shape=[num_clock, n], dtype=int)

        # run 1F1B
        for i in reversed(range(n)):
            # from the last device to the first
            for j in range(num_microbatches_remaining[i]):
                # running through all the remaining microbatches
                # forward
                next_clock = next_available_clock[i]
                schedules[next_clock][i] = (next_fwd_mb_idx[i], i)
                next_fwd_mb_idx[i] = next_fwd_mb_idx[i] + 1
                finished_bwd_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
                next_clock = next_clock + 1

                # backward
                # first, offset the next available clock to the clock
                # when the previous stage has just finished backward of the target mb.
                if i + 1 < n:  # not the last device
                    # find the next possible backward clock
                    while finished_bwd_batch_indices[next_clock][i + 1] <= next_bwd_mb_idx[i]:
                        assert finished_bwd_batch_indices[next_clock - 1][i] == next_bwd_mb_idx[i]
                        finished_bwd_batch_indices[next_clock][i] = finished_bwd_batch_indices[next_clock - 1][i]
                        next_clock = next_clock + 1

                schedules[next_clock][i] = (next_bwd_mb_idx[i], 2 * n - 1 - i)
                finished_bwd_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
                next_bwd_mb_idx[i] = next_bwd_mb_idx[i] + 1
                next_available_clock[i] = next_clock + 1


        # run cooldown passes
        for i in reversed(range(n)):
            for j in range(num_warmup_microbatches[i]):
                assert i + 1 < n
                next_clock = next_available_clock[i]
                while finished_bwd_batch_indices[next_clock][i + 1] <= next_bwd_mb_idx[i]:
                    finished_bwd_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
                    next_clock = next_clock + 1
                schedules[next_clock][i] = (next_bwd_mb_idx[i], 2 * n- 1 - i)
                finished_bwd_batch_indices[next_clock][i] = next_bwd_mb_idx
                next_bwd_mb_idx[i] = next_bwd_mb_idx[i] + 1
                next_available_clock = next_clock + 1
        return schedules

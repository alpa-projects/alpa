"""Generate pipeline schedules."""
import itertools
import logging
from abc import abstractmethod, ABCMeta
from typing import List, Tuple

import numpy as np

from alpa.pipeline_parallel.computation import PipelineComputation
from alpa.util import cached_property, OrderedSet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def gen_dependency_with_stages(
    compute_stages: List[PipelineComputation],
    apply_grad_stages: List[PipelineComputation] = ()):
    """Generate the dependency matrix for a list of pipeline stages."""
    n_stages = len(compute_stages) + len(apply_grad_stages)
    d = np.zeros([n_stages, n_stages], dtype=int)
    var_stage_id = {}
    for i, stage in enumerate(itertools.chain(compute_stages,
                                              apply_grad_stages)):
        for var in stage.invars:
            if var in var_stage_id:
                d[i, var_stage_id[var]] = 1
            else:
                # Assume the var is from global_invars
                pass
        for var in stage.outvars:
            var_stage_id[var] = i

    return d


def gen_linear_pipeline_dependency(num_stage):
    """
    Generate a dependency matrix.

    The matrix marks forward/backward stage pairs as neighbors. For test only.
    """
    assert num_stage % 2 == 0
    d = np.zeros([num_stage, num_stage], dtype=int)
    for i in range(num_stage - 1):
        d[i + 1][i] = 1
    for i in range(num_stage // 2):
        d[num_stage - 1 - i][i] = 1
    return d


class PipelineSchedule(metaclass=ABCMeta):
    """
    A pipeline schedule used by the distributed runtime.

    The core interface of this schedule is .schedule object.

    Args:
        dependency (np.array): dependency adjacency matrix.
        sliced_mesh (List[VirtualPhysicalMesh]): a list of pre-sliced virtual
            meshes to assign stages on.
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

        self._schedules: List[List[Tuple]] = self._generate_schedule()

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @abstractmethod
    def _generate_schedule(self):
        """Implementation of the schedule."""
        raise NotImplementedError()

    def pprint_schedule(self, to_print=False):
        """Pretty print the schedule."""
        printout = "\n"
        device_str = " ".join([f"d{d:<8}" for d in range(self.num_mesh)])
        printout = printout + f"Clock k : {device_str} \n"
        for clock, scheds in enumerate(self.schedules):
            sched_str = " ".join([f"{str(sched):<8}" for sched in scheds])
            printout = printout + f"Clock {clock:<2}: {sched_str} \n"
        if to_print:
            logger.info(printout)
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
        """Return the number of meshes."""
        return len(self.meshes)

    @property
    def num_clock(self):
        """Return the number of clocks in the schedule."""
        return len(self._schedules)

    @cached_property
    def stage_mesh_mapping(self):
        """Generate a stage-worker mapping according to the schedule."""
        placements = {}
        for tasks in self._schedules:
            for mesh_idx, task in enumerate(tasks):
                if task:
                    _, stage_idx = task
                    if stage_idx not in placements:
                        placements[stage_idx] = OrderedSet()
                    if mesh_idx not in placements[stage_idx]:
                        placements[stage_idx].add(mesh_idx)
        return placements

    @cached_property
    def mesh_stage_mapping(self):
        """Generate a worker-stage mapping according to the schedule."""
        ownership = {}
        for tasks in self._schedules:
            for mesh_idx, task in enumerate(tasks):
                if task:
                    _, stage_idx = task
                    if mesh_idx not in ownership:
                        ownership[mesh_idx] = OrderedSet()
                    if stage_idx not in ownership[mesh_idx]:
                        ownership[mesh_idx].add(stage_idx)
        return ownership

    def stage_placement(self, stage_idx):
        """Query the placement of a stage given its stage index."""
        return self.stage_mesh_mapping[stage_idx]

    def mesh_placement(self, mesh_idx):
        """Query the responsible stages of a worker given a worker index."""
        return self.mesh_stage_mapping[mesh_idx]

    def should_skip_grad_sync(self, task):
        """
        Query if grad sync (w/ other date replicas) should be skipped on a task.

        Args:
            task (Tuple[int]): (batch index, stage index).
        """
        batch_idx, _ = task
        return batch_idx != self.last_backward_batch_index

    @abstractmethod
    def previous_backward_batch_index(self, batch_idx):
        """Return microbatch index during backward prior to batch_idx."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def first_backward_batch_index(self):
        """Return the index of the first microbatch at backward pass."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def last_backward_batch_index(self):
        """Return the index of the last microbatch at backward pass."""
        raise NotImplementedError()


class GpipeSchedule(PipelineSchedule):
    """Construct a Gpipe-like schedule."""

    @property
    def name(self):
        return "gpipe"

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
                    rev.append((m - 1 - task[0], 2 * n - 1 - task[1]))
                    # rev.append((task[0], 2 * n - 1 - task[1]))
            return rev

        # backward schedules
        # Note: large microbatch index is executed earlier in backward now.
        for k in range(num_clock):
            mapped_scheds = schedules[num_clock - k - 1]
            schedules.append(reverse(mapped_scheds))

        # apply_grad schedules
        scheds = [None] * n
        for stage_idx, worker in self.apply_grad_placement.items():
            scheds[worker] = (self.last_backward_batch_index, stage_idx)
        schedules.append(scheds)
        return schedules

    @property
    def first_backward_batch_index(self):
        """Return the index of the first microbatch at backward pass."""
        return 0
        # return self.num_batch - 1

    @property
    def last_backward_batch_index(self):
        """Return the index of the last microbatch at backward pass."""
        return self.num_batch - 1
        # return 0

    def previous_backward_batch_index(self, batch_idx):
        """Return the index of the previous microbatch at backward pass."""
        assert batch_idx > 0
        return batch_idx - 1
        # return batch_idx + 1


class PipeDreamFlush(PipelineSchedule):
    """
    Generate a PipeDream-Flush schedule (a.k.a. 1F1B).

    It has similar latency to GPipe but is more memory-efficient.
    """

    @property
    def name(self):
        return "1f1b"

    def _generate_schedule(self):
        m = self.num_batch
        n = self.num_mesh

        # equal to gpipe
        num_clock = (m + n - 1) * 2
        schedules = [[None] * n for k in range(num_clock)]

        num_warmup_microbatches = [min(n - i - 1, m) for i in range(n)]
        num_microbatches_remaining = [m - i for i in num_warmup_microbatches]

        next_fwd_mb_idx = [0 for _ in range(n)]
        next_bwd_mb_idx = [0 for _ in range(n)]
        next_available_clock = list(range(n))
        finished_bwd_batch_indices = np.zeros(shape=[num_clock, n],
                                              dtype=np.int32)

        # warm-up clocks
        for i in range(n):
            for _ in range(num_warmup_microbatches[i]):
                schedules[next_available_clock[i]][i] = (next_fwd_mb_idx[i], i)
                next_available_clock[i] = next_available_clock[i] + 1
                next_fwd_mb_idx[i] = next_fwd_mb_idx[i] + 1

        # run 1F1B
        for i in reversed(range(n)):
            # from the last device to the first
            for _ in range(num_microbatches_remaining[i]):
                # running through all the remaining microbatches
                # forward
                next_clock = next_available_clock[i]
                schedules[next_clock][i] = (next_fwd_mb_idx[i], i)
                next_fwd_mb_idx[i] = next_fwd_mb_idx[i] + 1
                finished_bwd_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
                next_clock = next_clock + 1

                # backward
                # first, offset the next available clock to the clock
                # when the previous stage has just finished backward of the
                # target mb.
                if i + 1 < n:  # not the last device
                    # find the next possible backward clock
                    while finished_bwd_batch_indices[next_clock][
                            i + 1] <= next_bwd_mb_idx[i]:
                        assert finished_bwd_batch_indices[
                            next_clock - 1][i] == next_bwd_mb_idx[i]
                        finished_bwd_batch_indices[next_clock][
                            i] = finished_bwd_batch_indices[next_clock - 1][i]
                        next_clock = next_clock + 1

                schedules[next_clock][i] = (next_bwd_mb_idx[i], 2 * n - 1 - i)
                finished_bwd_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
                next_bwd_mb_idx[i] = next_bwd_mb_idx[i] + 1
                next_available_clock[i] = next_clock + 1

        # run cooldown passes
        for i in reversed(range(n)):
            for _ in range(num_warmup_microbatches[i]):
                assert i + 1 < n
                next_clock = next_available_clock[i]
                while finished_bwd_batch_indices[next_clock][
                        i + 1] <= next_bwd_mb_idx[i]:
                    finished_bwd_batch_indices[next_clock][i] = next_bwd_mb_idx[
                        i]
                    next_clock = next_clock + 1
                schedules[next_clock][i] = (next_bwd_mb_idx[i], 2 * n - 1 - i)
                finished_bwd_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
                next_bwd_mb_idx[i] = next_bwd_mb_idx[i] + 1
                next_available_clock[i] = next_clock + 1
            # update status matrix for the last worker
            if i > 0:
                finished_bwd_batch_indices[next_available_clock[i]:num_clock,
                                           i] = m

        # append apply_grad schedules
        scheds = [None] * n
        for stage_idx, worker in self.apply_grad_placement.items():
            scheds[worker] = (self.last_backward_batch_index, stage_idx)
        schedules.append(scheds)
        return schedules

    @property
    def first_backward_batch_index(self):
        """Return the index of the first microbatch at backward pass."""
        return 0

    @property
    def last_backward_batch_index(self):
        """Return the index of the last microbatch at backward pass."""
        return self.num_batch - 1

    def previous_backward_batch_index(self, batch_idx):
        """Return the index of the previous microbatch at backward pass."""
        assert batch_idx > 0
        return batch_idx - 1


class InferenceSchedule(PipelineSchedule):
    """Construct a Gpipe-like schedule."""

    @property
    def name(self):
        return "inference"

    def _generate_schedule(self):
        """
        Generate a forward-only schedule.

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

        # There should be no apply_grad tasks in the inference schedule.
        # apply_grad schedules
        scheds = [None] * n
        for stage_idx, worker in self.apply_grad_placement.items():
            scheds[worker] = (self.last_backward_batch_index, stage_idx)
        schedules.append(scheds)

        return schedules

    @property
    def first_backward_batch_index(self):
        """Return the index of the first microbatch at backward pass."""
        return 0

    @property
    def last_backward_batch_index(self):
        """Return the index of the last microbatch at backward pass."""
        return self.num_batch - 1

    def previous_backward_batch_index(self, batch_idx):
        """Return the index of the previous microbatch at backward pass."""
        assert batch_idx > 0
        return batch_idx - 1

import functools
import math

import numpy as np


def cached_property(fn, *args, **kwargs):
    """
    Decorator to make a function a "cached property".

    This means that it is a property whose return value is cached after the
    first time it is called.

    Args:
        fn: The function to be made a cached property
        *args: Any args for the function
        **kwargs: Any kwargs for the function
    Returns:
        function
    """
    return property(functools.lru_cache()(fn, *args, **kwargs))


def gen_linear_dependency(num_stage):
    """Generate a linear dependency matrix."""
    d = np.zeros([num_stage, num_stage], dtype=np.int)
    for i in range(num_stage - 1):
        d[i + 1][i] = 1
    return d


def gen_linear_pipeline_dependency(num_stage):
    """
    Generate a dependency matrix that marks the neighbors and forward/backward
    stage pairs as neighbors.
    """
    assert num_stage % 2 == 0
    d = np.zeros([num_stage, num_stage], dtype=np.int)
    for i in range(num_stage - 1):
        d[i + 1][i] = 1
    for i in range(num_stage // 2):
        d[num_stage - 1 - i][i] = 1
    return d


def gen_linear_pipeline_dependency_with_apply(num_stage, mesh_num, apply_deps):
    """
    Generate dependency matrix marks compute grad and apply grad
    """
    d = np.zeros((num_stage, num_stage), dtype=np.int32)
    for i in range(mesh_num * 2 - 1):
        d[i + 1][i] = 1
    for i in range(mesh_num):
        d[mesh_num * 2 - 1 - i][i] = 1
    for pair in apply_deps:
        d[pair[0]][pair[1]] = 1
    return d


class GpipeSchedule:
    """
    Construct a Gpipe-like schedule.

    Args:
        dependency (np.array): dependency adjacency matrix.
        mesh (VirtualMesh): a virtual mesh representing the entire cluster.
        sliced_mesh (List[VirtualMesh]): a list of pre-sliced virtual meshes
            to assign workers on.
        num_pipeline_worker (int):
        apply_grad_schedule (Dict[int, int]): A map from apply grad's stage idx
            to the worker it is assigned
        num_batch (int): number of microbatches.
        costs (List[int]): running costs of each stage.
    """

    def __init__(self,
                 *,
                 dependency,
                 mesh,
                 num_pipeline_worker,
                 apply_grad_schedule,
                 sliced_meshes=None,
                 num_batch=1,
                 costs=None):
        self.dependency = dependency
        self.original_mesh = mesh
        self.meshes = sliced_meshes
        self.apply_grad_schedule = apply_grad_schedule
        self.num_batch = num_batch
        self.costs = costs
        self.num_stage = dependency.shape[0]

        self.num_pipeline_worker = num_pipeline_worker
        # TODO (zhuohan): Seperate device placement and runtime scheduling
        if not self.meshes:
            # These are virtual meshes
            self.meshes = self.slice_mesh(self.original_mesh)
        if len(self.meshes) != self.num_pipeline_worker:
            raise RuntimeError("Gpipe schedule requires #meshes = #workers.")
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
        n = self.num_pipeline_worker
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
        # apply grad schedules
        scheds = [None] * n
        for stage_idx, worker in self.apply_grad_schedule.items():
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
                        placements[stage_idx] = set()
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
                        ownership[worker_idx] = set()
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

    def __len__(self):
        return len(self._schedules)

    @property
    def num_clock(self):
        """Return the number of clocks in the schedule."""
        return len(self._schedules)

    @property
    def num_worker(self):
        """Return the number of workers (physical meshes)."""
        return self.num_pipeline_worker

    @property
    def num_mesh(self):
        """Return the number of meshes in the schedule."""
        return self.num_pipeline_worker

    def slice_mesh(self, original_mesh):
        """
        Slice the mesh for each remote runner.

        In this impl, we guarantee the slicing follows:
        - len(sliced_meshes) == num_stages / 2 (place forward/backward in a mesh);
        - higher priority to slice over the node dimension rather than gpu dimension.

        Args:
            original_mesh: a virtual device mesh.

        Returns:
            sliced_meshes (List[Mesh]): List of meshes to spawn worker on.
        """
        output_meshes = []
        num_mesh_expected = self.num_pipeline_worker
        if not original_mesh.is_distributed:
            raise RuntimeError("SingleDeviceMesh is not supported.")
        if original_mesh.total_devices < num_mesh_expected:
            raise RuntimeError("#device < #workers.")

        num_device_per_mesh = int(original_mesh.total_devices /
                                  num_mesh_expected)
        num_device_per_host = original_mesh.num_devices_per_host
        num_host = original_mesh.num_hosts
        if num_device_per_host >= num_device_per_mesh:
            num_mesh_per_host = num_device_per_host // num_device_per_mesh
            for i in range(num_mesh_expected):
                host_idx = i // num_mesh_per_host
                mesh_idx = i % num_mesh_per_host
                ind = list(range(num_device_per_host))
                mesh = original_mesh.slice(0, [host_idx])\
                    .slice(1, [ind[mesh_idx * num_device_per_mesh:(mesh_idx + 1) * num_device_per_mesh]])
                output_meshes.append(mesh)
        else:
            num_host_per_mesh = math.ceil(num_device_per_mesh /
                                          num_device_per_host)
            ind = list(range(num_host))
            for i in range(num_mesh_expected):
                output_meshes.append((original_mesh.slice(
                    0, ind[num_host_per_mesh * i:num_host_per_mesh * (i + 1)])))
        return output_meshes

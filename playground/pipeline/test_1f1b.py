import numpy as np

from parax.pipeline_parallel.schedules import GpipeSchedule, PipeDreamFlush


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


def test_gpipe_schedule(num_stage, num_mesh, num_batch):
    deps = gen_linear_pipeline_dependency(num_stage)
    meshes = [None] * num_mesh
    apply_placements = dict()
    s = GpipeSchedule(dependency=deps,
                      meshes=meshes,
                      apply_grad_placement=apply_placements,
                      num_batch=num_batch)
    s.pprint_schedule()


def test_1f1b_schedule(num_stage, num_mesh, num_batch):
    deps = gen_linear_pipeline_dependency(num_stage)
    meshes = [None] * num_mesh
    apply_placements = dict()
    s = PipeDreamFlush(dependency=deps,
                       meshes=meshes,
                       apply_grad_placement=apply_placements,
                       num_batch=num_batch)
    s.pprint_schedule()


num_stage = 8
num_mesh = 4
num_batch = 8
test_gpipe_schedule(num_stage, num_mesh, num_batch)
test_1f1b_schedule(num_stage, num_mesh, num_batch)
import unittest

from alpa.pipeline_parallel.schedules import (gen_linear_pipeline_dependency,
                                              GpipeSchedule, PipeDreamFlush)


class PipelineScheduleTest(unittest.TestCase):

    def run_schedule_basics(self, schedule_type, num_stage, num_mesh,
                            num_batch):
        deps = gen_linear_pipeline_dependency(num_stage)
        meshes = [None] * num_mesh
        num_fwd_stage = num_stage // 2
        apply_grad_placement = {num_stage + i: i for i in range(num_fwd_stage)}
        if schedule_type == "gpipe":
            schedule_cls = GpipeSchedule
        elif schedule_type == "1f1b":
            schedule_cls = PipeDreamFlush
        else:
            print("unrecognized type of schedule.")
            return

        s = schedule_cls(dependency=deps,
                         meshes=meshes,
                         apply_grad_placement=apply_grad_placement,
                         num_batch=num_batch)

        # check num_clock
        assert s.num_clock == (num_mesh + num_batch - 1) * 2 + 1, (
            "clock number wrong.")

        # check no stage is on > 1 meshes
        for i in range(num_stage):
            mesh_indices = s.stage_placement(i)
            assert len(mesh_indices) == 1, (
                "we only support each stage placed on one mesh.")

        # check no mesh owns > 3 stages (forward, backward, apply_grad)
        for i in range(num_mesh):
            stage_indices = s.mesh_placement(i)
            assert len(stage_indices) == 3, (
                "One mesh at most owns three stages: forward, backward,"
                " and apply_grad stages.")
            stage_indices_list = list(stage_indices)
            stage_indices_list.sort()
            f, b, a = stage_indices_list[0], stage_indices_list[
                1], stage_indices_list[2]
            assert f == 2 * num_mesh - 1 - b
            assert a == num_stage + f

    def run_1f1b(self, num_stage, num_mesh, num_batch):
        deps = gen_linear_pipeline_dependency(num_stage)
        meshes = [None] * num_mesh
        num_fwd_stage = num_stage // 2
        apply_grad_placement = {num_stage + i: i for i in range(num_fwd_stage)}
        s = PipeDreamFlush(dependency=deps,
                           meshes=meshes,
                           apply_grad_placement=apply_grad_placement,
                           num_batch=num_batch)

        # test the in-flight microbatches <= num_mesh
        in_flight = [0 for _ in range(num_mesh)]
        max_in_flight = [0 for _ in range(num_mesh)]
        for sched in s.schedules:
            for mesh_idx, task in enumerate(sched):
                if task:
                    batch_idx, stage_idx = task
                    if stage_idx < num_stage / 2:
                        in_flight[mesh_idx] += 1
                    if stage_idx < num_stage and stage_idx >= num_stage / 2:
                        in_flight[mesh_idx] -= 1
                    if in_flight[mesh_idx] > max_in_flight[mesh_idx]:
                        max_in_flight[mesh_idx] = in_flight[mesh_idx]

        for i in range(num_mesh):
            assert max_in_flight[i] <= num_mesh - i, (
                "max number of in-flight is incorrect.")

    def test_schedules(self):
        schedule_types = ["gpipe", "1f1b"]
        num_stages = [4, 6, 8, 12, 16, 32, 64]
        num_batches = [1, 2, 4, 8, 16, 32, 64, 128]
        for schedule_type in schedule_types:
            for num_stage in num_stages:
                for num_batch in num_batches:
                    num_mesh = num_stage // 2
                    #print(
                    #    "Testing case: type {}, num_stage {}, num_mesh {}, num_batch {}."
                    #    .format(schedule_type, num_stage, num_mesh, num_batch))
                    self.run_schedule_basics(schedule_type, num_stage, num_mesh,
                                             num_batch)
                    if schedule_type == "1f1b":
                        self.run_1f1b(num_stage, num_mesh, num_batch)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineScheduleTest("test_schedules"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

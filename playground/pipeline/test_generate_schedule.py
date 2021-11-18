"""Experimental code to generate a Gpipe clock-cycle schedule."""
import numpy as np


def generate_gpipe_schedule(m, n):
    num_clock = m + n - 1
    schedules = []
    for k in range(num_clock):
        scheds = [None] * n
        for d in range(max(1 + k - m, 0), min(1 + k, n)):
            scheds[d] = (k - d, d)
        schedules.append(scheds)

    def reverse(scheds):
        reversed = []
        for task in scheds:
            if not task:
                reversed.append(None)
            else:
                reversed.append((m - 1 - task[0], 2 * n - 1 - task[1]))
        return reversed

    # backward schedules
    for k in range(num_clock):
        mapped_scheds = schedules[num_clock - k - 1]
        schedules.append(reverse(mapped_scheds))
    return schedules


def generate_1f1b_schedule(m, n):
    # equal to gpipe
    num_clock = (m + n - 1) * 2
    schedules = [[None] * n for k in range(num_clock)]

    num_warmup_microbatches = [min(n - i - 1, m) for i in range(n)]
    num_microbatches_remaining = [m - i for i in num_warmup_microbatches]

    next_fwd_mb_idx = [0 for _ in range(n)]
    next_bwd_mb_idx = [0 for _ in range(n)]
    next_available_clock = [i for i in range(n)]
    finished_bwd_batch_indices = np.zeros(shape=[num_clock, n], dtype=np.int32)

    # warm-up clocks
    for i in range(n):
        for j in range(num_warmup_microbatches[i]):
            schedules[next_available_clock[i]][i] = (next_fwd_mb_idx[i], i)
            next_available_clock[i] = next_available_clock[i] + 1
            next_fwd_mb_idx[i] = next_fwd_mb_idx[i] + 1

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
            finished_bwd_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
            next_bwd_mb_idx[i] = next_bwd_mb_idx[i] + 1
            next_available_clock[i] = next_clock + 1
        # update status matrix for the last worker
        if i > 0:
            finished_bwd_batch_indices[next_available_clock[i]:num_clock, i] = m

    return schedules



def pprint_schedule(schedules):
    num_device = len(schedules[0])
    device_str = " ".join(["{:<8}".format("d" + str(d)) for d in range(num_device)])
    print("Clock {:<2}: {}".format("id", device_str))
    for clock, scheds in enumerate(schedules):
        sched_str = " ".join(["{:<8}".format(str(sched)) for sched in scheds])
        print("Clock {:<2}: {}".format(clock, sched_str))


if __name__ == "__main__":
    m = 4
    n = 3
    schedules = generate_gpipe_schedule(m, n)
    pprint_schedule(schedules)
    print("\n")
    schedules = generate_1f1b_schedule(m, n)
    pprint_schedule(schedules)

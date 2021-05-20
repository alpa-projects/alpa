"""Experimental code to generate a Gpipe clock-cycle schedule."""
def generate_schedule(m, n):
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
                reversed.append((task[0], 2 * n - 1 - task[1]))
        return reversed

    # backward schedules
    for k in range(num_clock):
        mapped_scheds = schedules[num_clock - k - 1]
        schedules.append(reverse(mapped_scheds))
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
    schedules = generate_schedule(m, n)
    pprint_schedule(schedules)

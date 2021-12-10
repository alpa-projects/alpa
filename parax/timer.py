"""Global timer for profiling."""
import time

do_sync = True


class _Timer:
    """An internal timer."""

    def __init__(self, name):
        self.name = name
        self.started = False
        self.start_time = None
        self.costs = []

        # Loop timer
        self.ever_suspended = False
        self.accum_cost = 0.0

    def start(self, sync_func=None):
        """Start the timer."""
        assert not self.started, "timer has already been started"
        if sync_func and do_sync:
            sync_func()
        self.start_time = time.time()
        self.started = True

    def suspend(self, sync_func=None):
        """Suspend the timer in a loop."""
        assert self.started
        self.ever_suspended = True

        # we accumulate on the accum_cost
        if sync_func and do_sync:
            sync_func()
        self.accum_cost += time.time() - self.start_time
        self.started = False

    def stop(self, sync_func=None):
        """Stop the timer."""
        if self.ever_suspended:
            assert not self.started, f"Stop the timer {self.name} before suspending it."
        else:
            assert self.started, f"timer {self.name} is not started nor ever suspended."
        if sync_func and do_sync:
            sync_func()
        if self.ever_suspended:
            self.costs.append(self.accum_cost)
            self.accum_cost = 0.0
        else:
            cost = time.time() - self.start_time
            self.costs.append(cost)
        self.started = False

    def reset(self):
        """Reset timer."""
        self.costs = []
        self.accum_cost = 0.0
        self.started = False
        self.ever_suspended = False

    def elapsed(self, mode="average"):
        """Calculate the elapsed time."""
        if not self.costs:
            raise RuntimeError("Empty costs.")
        if mode == "average":
            return sum(self.costs) / len(self.costs)
        elif mode == "sum":
            return sum(self.costs)
        else:
            raise RuntimeError("Supported mode is: average | sum")

    def log(self, mode="average", normalizer=1.0):
        """Log a timer's cost in different modes."""
        assert normalizer > 0.0
        string = "time (ms)"

        elapsed = self.elapsed(mode) * 1000.0 / normalizer
        string += ' | {}: {:.2f}'.format(self.name, elapsed)
        print(string, flush=True)


class Timers:
    """A group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def __contains__(self, name):
        return name in self.timers

    def log(self, names, normalizer=1.0):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed() * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        print(string, flush=True)


timers = Timers()

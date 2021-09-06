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

    def start(self, sync_func=None):
        """Start the timer."""
        assert not self.started, "timer has already been started"
        if sync_func and do_sync:
            sync_func()
        self.start_time = time.time()
        self.started = True

    def stop(self, sync_func=None):
        """Stop the timer."""
        assert self.started, "timer is not started"
        if sync_func and do_sync:
            sync_func()
        self.costs.append(time.time() - self.start_time)
        self.started = False

    def reset(self):
        """Reset timer."""
        self.costs = []
        self.started = False


class Timers:
    """A group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        print(string, flush=True)


timers = Timers()

import asyncio
import heapq
from collections import deque, OrderedDict


class WeightedRoundRobin:
    """
    Scheduler that cycles between queues of different weightings.
    The interface is the same as it were a queue implemented using deque().
    This implementation extends the original algorithm by allowing non-integer
    priorities. All weights in this class are implicitly divided by a scale
    factor - if all the queue weights are integer multiples of the scale
    factor, the algorithm behaves just like standard weighted round robin.
    Using smaller weights makes the scheduler switch between queues more
    frequently, improving latency.
    """
    # The scheduling algorithm is implemented using an event list. Each queue
    # is associated with an hourglass that fills up a certain fraction every
    # time step. When the hourglass is filled, a task is scheduled from the
    # corresponding queue. An hourglass is allowed to be filled faster than
    # 100% per time step - in this case, tasks are consecutively scheduled
    # from the same queue until the hourglass is no longer full.

    class Hourglass:
        def __init__(self, update_time, amnt_filled):
            self.update_time = update_time
            self.amnt_filled = amnt_filled
            self.linked_tasks = deque()

        def __repr__(self):
            return '({}, {}, {})'.format(
                self.update_time, self.amnt_filled, list(self.linked_tasks))

    def __init__(self, weights, scale, default_weight=None,
                 max_empty_hourglasses=100):
        self.weights = weights
        self.default_weight = default_weight
        self.scale = scale
        self.max_empty_hourglasses = max_empty_hourglasses
        self.curr_item_num = 0
        self.curr_simulated_time = 0
        self.tasks = {}
        self.hourglasses = {}
        self.event_list = []
        self.empty_hourglasses = OrderedDict()

    def __len__(self):
        return len(self.tasks)

    def append(self, name_and_item):
        queue_name, item = name_and_item
        self.tasks[self.curr_item_num] = item
        new_event = False
        if queue_name in self.empty_hourglasses:
            self.hourglasses[queue_name] = self.empty_hourglasses[queue_name]
            del self.empty_hourglasses[queue_name]
            new_event = True
        if queue_name not in self.hourglasses:
            self.hourglasses[queue_name] = \
                WeightedRoundRobin.Hourglass(0, 0)
            new_event = True
        hourglass = self.hourglasses[queue_name]
        hourglass.linked_tasks.append(self.curr_item_num)
        if new_event:
            hourglass.update_time = self.curr_simulated_time
            self.__add_new_event(hourglass, queue_name)
        self.curr_item_num += 1

    def extend(self, items):
        for item in items:
            self.append(item)

    def popleft(self):
        event_entry = heapq.heappop(self.event_list)
        queue_name = event_entry[2]
        hourglass = self.hourglasses[queue_name]
        if hourglass.amnt_filled >= self.scale:
            hourglass.amnt_filled -= self.scale
        else:
            self.curr_simulated_time = event_entry[0]
            weight = self.weights.get(queue_name, self.default_weight)
            if weight is None:
                raise KeyError
            hourglass.amnt_filled += (
                self.curr_simulated_time - hourglass.update_time) * weight
            hourglass.amnt_filled -= self.scale
        hourglass.update_time = self.curr_simulated_time
        task_num = hourglass.linked_tasks.popleft()
        task = self.tasks.pop(task_num)
        if len(hourglass.linked_tasks) == 0:
            del self.hourglasses[queue_name]
            self.empty_hourglasses[queue_name] = hourglass
            if len(self.empty_hourglasses) > self.max_empty_hourglasses:
                self.empty_hourglasses.popitem(last=False)
        else:
            self.__add_new_event(hourglass, queue_name)
        return (queue_name, task)

    def __add_new_event(self, hourglass, queue_name):
        if hourglass.amnt_filled >= self.scale:
            event_time = self.curr_simulated_time
            event_entry = (event_time, hourglass.linked_tasks[0], queue_name)
            heapq.heappush(self.event_list, event_entry)
        else:
            weight = self.weights.get(queue_name, self.default_weight)
            if weight is None:
                raise KeyError
            time_to_full = (
                self.scale - hourglass.amnt_filled + weight - 1) // weight
            event_time = self.curr_simulated_time + time_to_full
            event_entry = (event_time, hourglass.linked_tasks[0], queue_name)
            heapq.heappush(self.event_list, event_entry)

    def verify_state(self):
        """Checks the invariants of the class"""
        task_nums = []
        try:
            assert len(self.event_list) == 0 or \
                self.curr_simulated_time <= self.event_list[0][0]
            for queue_name, hourglass in self.hourglasses.items():
                assert len(hourglass.linked_tasks) > 0
                for task_num in hourglass.linked_tasks:
                    assert task_num in self.tasks
                assert hourglass.amnt_filled >= 0
                assert queue_name not in self.empty_hourglasses
                task_nums += list(hourglass.linked_tasks)
                if hourglass.amnt_filled >= self.scale:
                    assert self.event_list[0][0] == self.curr_simulated_time
                    assert self.curr_simulated_time == hourglass.update_time
            for hourglass in self.empty_hourglasses.values():
                assert len(hourglass.linked_tasks) == 0
                assert hourglass.amnt_filled >= 0
            assert sorted(task_nums) == sorted(list(self.tasks.keys()))
        except AssertionError as e:
            e.args += (repr(self),)
            raise e

    def __repr__(self):
        return "Tasks: {}\nEvent list: {}\nHourglasses: {}\nTime: {}".format(
            self.tasks, self.event_list, self.hourglasses,
            self.curr_simulated_time)


class NestedScheduler:
    """
    Scheduler where each queue is an independent inner scheduler object.
    This can be used to implement hierarchies of weights and queues.
    """
    def __init__(self, outer_scheduler, inner_schedulers):
        self.outer_scheduler = outer_scheduler
        self.inner_schedulers = inner_schedulers

    def __len__(self):
        return len(self.outer_scheduler)

    def append(self, name_and_item):
        name, item = name_and_item
        self.outer_scheduler.append((name, None))
        self.inner_schedulers[name].append(item)

    def extend(self, items):
        for item in items:
            self.append(item)

    def popleft(self):
        name = self.outer_scheduler.popleft()[0]
        return (name, self.inner_schedulers[name].popleft())

    def __repr__(self):
        return '\n'.join(
            ['Outer: ' + repr(self.outer_scheduler)] +
            [repr(name) + ': ' + repr(s)
             for (name, s) in self.inner_schedulers.items()])


class FrontQueueScheduler:
    """
    Scheduler decorator that allows tasks to be placed at the front of the
    queue. The front behaves like the front of a deque(), i.e. it is LIFO.
    """
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.front_queue = deque()

    def __len__(self):
        return len(self.front_queue) + len(self.scheduler)

    def append(self, item):
        self.scheduler.append(item)

    def extend(self, items):
        for item in items:
            self.append(item)

    def popleft(self):
        if len(self.front_queue) > 0:
            return self.front_queue.popleft()
        return self.scheduler.popleft()

    def appendleft(self, item):
        self.front_queue.appendleft(item)

    def extendleft(self, items):
        self.front_queue.extendleft(items)

    def __repr__(self):
        return "Front queue:{}\n{}".format(self.front_queue, self.scheduler)


class AsyncWrapper:
    """
    Decorator that makes a scheduler object behave like an asyncio.Queue().
    """
    def __init__(self, scheduler):
        self.schedule_waitlist = asyncio.Queue()
        self.scheduler = scheduler

    @property
    def maxsize(self):
        return 0

    def qsize(self):
        return len(self.scheduler) + self.schedule_waitlist.qsize()

    def empty(self):
        return len(self.scheduler) == 0 and self.schedule_waitlist.empty()

    def full(self):
        return False

    async def put(self, item):
        await self.schedule_waitlist.put((item, None))

    def put_nowait(self, item):
        self.schedule_waitlist.put_nowait((item, None))

    async def get(self):
        if self.empty():
            self.__process_waitlist_item(await self.schedule_waitlist.get())
        while not self.schedule_waitlist.empty():
            self.__process_waitlist_item(
                self.schedule_waitlist.get_nowait())
        return self.scheduler.popleft()

    def get_nowait(self):
        if self.empty():
            raise asyncio.QueueEmpty
        while not self.schedule_waitlist.empty():
            self.__process_waitlist_item(self.schedule_waitlist.get_nowait())
        return self.scheduler.popleft()

    def __process_waitlist_item(self, waitlist_item):
        data, strategy = waitlist_item
        if strategy is None:
            self.scheduler.append(data)
        else:
            strategy(self.scheduler, data)

    def task_done(self):
        self.scheduler_waitlist.task_done()

    async def join(self):
        await self.scheduler_waitlist.join()

    def put_nowait_special(self, strategy, data):
        """Must add exactly one item into the schedule"""
        self.schedule_waitlist.put_nowait((data, strategy))

    def __repr__(self):
        return repr(self.scheduler)

"""Abstract class for collective groups."""
from abc import ABCMeta
from abc import abstractmethod
import logging
import datetime
import time

import ray

from alpa.collective.const import get_store_name
from alpa.collective.types import (AllReduceOptions, BarrierOptions,
                                   ReduceOptions, AllGatherOptions,
                                   BroadcastOptions, ReduceScatterOptions)

logger = logging.getLogger(__name__)


class Rendezvous:
    """A rendezvous class for different actor/task processes to meet.

    To initialize an NCCL collective communication group, different
    actors/tasks spawned in Ray in a collective group needs to meet
    each other to synchronize the NCCLUniqueID. This class guarantees
    they meet via the NCCLUniqueIDStore, initialized on the rank=0
    process.

    Args:
        store_key (str): the unique store key, usually as a concatanation
            of group_name and communicator key. See `get_nccl_communicator`
            for more details.
    """

    def __init__(self, store_key):
        if not store_key:
            raise ValueError(
                "Invalid store_key. The store_key is a concatenation of "
                "'group_name' and the 'communicator_key'. See the "
                "docstring of `get_nccl_communicator` for details.")
        self._store_key = store_key
        self._store_name = None
        self._store = None

    def meet(self, timeout_s=180):
        """Meet at the named actor store.

        Args:
            timeout_s (int): timeout in seconds.

        Return:
            None
        """
        if timeout_s <= 0:
            raise ValueError("The 'timeout' argument must be positive. "
                             f"Got '{timeout_s}'.")
        self._store_name = get_store_name(self._store_key)
        timeout_delta = datetime.timedelta(seconds=timeout_s)
        elapsed = datetime.timedelta(seconds=0)
        start_time = datetime.datetime.now()
        while elapsed < timeout_delta:
            try:
                logger.debug(
                    f"Trying to meet at the store '{self._store_name}'")
                self._store = ray.get_actor(self._store_name)
            except ValueError:
                logger.debug(
                    f"Failed to meet at the store '{self._store_name}'. "
                    "Trying again...")
                time.sleep(1)
                elapsed = datetime.datetime.now() - start_time
                continue
            logger.debug("Successful rendezvous!")
            break
        if not self._store:
            raise RuntimeError("Unable to meet other processes "
                               "at the rendezvous store. If you are using "
                               "P2P communication, please check if tensors "
                               "are put in the correct GPU. ")

    @property
    def store(self):
        return self._store

    def get_nccl_id(self, timeout_s=180):
        """Get the NCCLUniqueID from the store through Ray.

        Args:
            timeout_s: timeout in seconds.

        Return:
            uid (str): the NCCLUniqueID if successful.
        """
        if not self._store:
            raise ValueError("Rendezvous store is not setup.")
        uid = None
        timeout_delta = datetime.timedelta(seconds=timeout_s)
        elapsed = datetime.timedelta(seconds=0)
        start_time = datetime.datetime.now()
        while elapsed < timeout_delta:
            uid = ray.get(self._store.get_id.remote())
            if not uid:
                time.sleep(1)
                elapsed = datetime.datetime.now() - start_time
                continue
            break
        if not uid:
            raise RuntimeError("Unable to get the NCCLUniqueID from the store.")
        return uid

    def get_access_counter(self):
        """Return how many times the NCCLUniqueID has been accessed."""
        return ray.get(self._store.get_access_counter.remote())

    def destroy_store(self):
        """Delete the named actor."""
        ray.kill(self._store)
        self._store = None


class BaseGroup(metaclass=ABCMeta):
    """Abstract class for collective groups."""

    def __init__(self, world_size, rank, group_name):
        """Init the process group with basic information.

        Args:
            world_size (int): The total number of processes in the group.
            rank (int): The rank of the current process.
            group_name (str): The group name.
        """
        self._world_size = world_size
        self._rank = rank
        self._group_name = group_name

    @property
    def rank(self):
        """Return the rank of the current process."""
        return self._rank

    @property
    def world_size(self):
        """Return the number of processes in this group."""
        return self._world_size

    @property
    def group_name(self):
        """Return the group name of this group."""
        return self._group_name

    @classmethod
    def backend(cls):
        """The backend of this collective group."""
        raise NotImplementedError()

    @abstractmethod
    def allreduce(self, tensors, allreduce_options=AllReduceOptions()):
        raise NotImplementedError()

    @abstractmethod
    def barrier(self, barrier_options=BarrierOptions()):
        raise NotImplementedError()

    @abstractmethod
    def reduce(self, tensors, reduce_options=ReduceOptions()):
        raise NotImplementedError()

    @abstractmethod
    def allgather(self,
                  tensor_lists,
                  tensors,
                  allgather_options=AllGatherOptions()):
        raise NotImplementedError()

    @abstractmethod
    def broadcast(self, tensors, broadcast_options=BroadcastOptions()):
        raise NotImplementedError()

    @abstractmethod
    def reducescatter(self,
                      tensors,
                      tensor_lists,
                      reducescatter_options=ReduceScatterOptions()):
        raise NotImplementedError()

    @abstractmethod
    def send(self, tensors, send_options):
        raise NotImplementedError()

    @abstractmethod
    def recv(self, tensors, recv_options):
        raise NotImplementedError()

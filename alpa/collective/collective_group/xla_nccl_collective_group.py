"""NCCL-based collective operations with apis from xla extension."""
import logging

import ray
from jax._src.lib import xla_extension as xe

from alpa.collective.collective_group import xla_nccl_util
from alpa.collective.collective_group.base_collective_group import BaseGroup, Rendezvous
from alpa.collective.const import get_store_name, ENV
from alpa.collective.types import (Backend, BroadcastOptions, AllReduceOptions,
                                   BarrierOptions, ReduceOptions,
                                   AllGatherOptions, ReduceScatterOptions,
                                   SendOptions, RecvOptions)

logger = logging.getLogger(__name__)


class XLANCCLGroup(BaseGroup):
    """NCCL-based collective operations with apis from xla extension."""

    def __init__(self, world_size, rank, group_name):
        """Init an NCCL collective group."""
        super().__init__(world_size, rank, group_name)

        # communicator and stream cache.
        # TODO (Hao): we need a lock here...
        self._barrier_tensor = None
        self._dev_comm_map = {}
        self._dev_streams_map = {}

        # record the used GPU IDs.
        self._used_gpu_indices = set()

        # TODO(Fu): might need an event map
        self._dev_event_map = {}

        if xla_nccl_util.get_nccl_runtime_version() < 2704:
            logger.warning("NCCL send/recv calls requires NCCL>=2.7.4")

    def destroy_group(self):
        """Destroy the group and release NCCL communicators."""
        if len(self._dev_comm_map.keys()) > 0:

            # Destroy the communicators and streams.
            for comm_key, comms in self._dev_comm_map.items():
                xe.nccl_destroy_comms(comms)
                self._dev_comm_map[comm_key] = None

        if self.rank == 0:
            for comm_key in self._dev_comm_map:
                assert not self._dev_comm_map[comm_key]
                group_key = self._generate_group_key(comm_key)
                self._destroy_store(group_key)
        self._barrier_tensor = None
        self._dev_comm_map = None
        self._dev_streams_map = None

    @classmethod
    def backend(cls):
        return Backend.NCCL

    def broadcast_partialgpu(self,
                             tensors,
                             broadcast_options=BroadcastOptions()):
        """Broadcast tensors to all other gpus following options.
        It will only involve subset of gpu in this worker.

        Args:
            tensors (List): tensors to be broadcast or received.
            broadcast_options: broadcast options.

        Returns:
            None
        """
        root_rank = 0

        key = broadcast_options.comm_key
        comms = self._get_nccl_broadcast_communicator(
            key, broadcast_options.world_size, broadcast_options.devices_ids,
            broadcast_options.devices_global_rank)
        xe.nccl_broadcast_partial_gpus(comms, tensors,
                                       broadcast_options.local_start_pos_list,
                                       broadcast_options.n_elements, root_rank)

    def _get_nccl_broadcast_communicator(self,
                                         comm_key,
                                         world_size,
                                         devices_ids,
                                         devices_global_rank,
                                         nccl_uid=None):
        """Create or retrieve a list of NCCL communicators for
        broadcast from cache. Here we only use partial devices in a host, so
        we create this function besides _get_nccl_collective_communicator.

        If the communicator is found in cache, return the communicator. If not,
        a communicator and a stream will be created and put in cache.

        Args:
            comm_key (str): the key to query the communicator cache.
            world_size (int): the number of devices in this collective
                              communicator.
            devices_ids (List): a list of GPU devices of the current process
                                that participates into the collective.
            devices_global_rank (List): the corresponding global rank for
                                device in devices_ids.
            nccl_uid : If it is None, we will create a nccl_uid here.

        Returns:
            communicator: the NCCL communicator corresponded to the devices.
        """
        if not comm_key:
            raise RuntimeError("Got empty communicator key.")

        for d in devices_ids:
            self._used_gpu_indices.add(d)

        # TODO(Hao): lock the _dev_comm_map here.
        if comm_key in self._dev_comm_map:
            return self._dev_comm_map[comm_key]

        group_key = self._generate_group_key(comm_key)
        if devices_global_rank[0] == 0:
            if nccl_uid is None:
                nccl_uid = self._generate_nccl_uid(group_key)
        else:
            if nccl_uid is None:
                rendezvous = Rendezvous(group_key)
                rendezvous.meet()
                nccl_uid = rendezvous.get_nccl_id()

                # Recycle the NCCLUniqueIDStore named actor *pro-activately* to
                # avoid named actor leak.
                if rendezvous.get_access_counter() == self.world_size:
                    logger.debug(
                        "NCCLUniqueID has been broadcasted. The "
                        "NCCLUniqueIDStore will go out of context and be "
                        "destroyed.")
                    rendezvous.destroy_store()

        comms = xe.nccl_create_communicators(world_size, devices_global_rank,
                                             devices_ids, nccl_uid,
                                             ENV.NCCL_USE_MULTISTREAM.val)
        self._dev_comm_map[comm_key] = comms
        return comms

    def send(self, tensors, send_options=SendOptions()):
        """Send a tensor to a destination gpu in the group.

        Args:
            tensors (List): the tensor to send.
            send_options: send options.

        Returns:
            None
        """

        buffer = tensors[0]
        my_gpu_idx = xe.get_buffer_device_id(buffer)
        peer_rank, peer_gpu_idx = \
            send_options.dst_rank, send_options.dst_gpu_index
        comm_key = _get_comm_key_send_recv(self.rank, my_gpu_idx, peer_rank,
                                           peer_gpu_idx)
        comms = self._get_nccl_p2p_communicator(comm_key, my_gpu_idx, peer_rank,
                                                peer_gpu_idx)

        peer_p2p_rank = 0 if self.rank > peer_rank else 1
        xe.nccl_send(comms, buffer, send_options.start_pos,
                     send_options.n_elements, peer_p2p_rank)

    def recv(self, tensors, recv_options=RecvOptions()):
        """Receive a tensor from a source gpu in the group.

        Args:
            tensors (List): the received tensor.
            recv_options: Receive options.

        Returns:
            None
        """

        buffer = tensors[0]
        my_gpu_idx = xe.get_buffer_device_id(buffer)
        peer_rank, peer_gpu_idx = \
            recv_options.src_rank, recv_options.src_gpu_index
        comm_key = _get_comm_key_send_recv(self.rank, my_gpu_idx, peer_rank,
                                           peer_gpu_idx)
        comms = self._get_nccl_p2p_communicator(comm_key, my_gpu_idx, peer_rank,
                                                peer_gpu_idx)

        peer_p2p_rank = 0 if self.rank > peer_rank else 1
        xe.nccl_recv(comms, buffer, recv_options.start_pos,
                     recv_options.n_elements, peer_p2p_rank)

    def _get_nccl_p2p_communicator(self,
                                   comm_key,
                                   my_gpu_idx,
                                   peer_rank,
                                   peer_gpu_idx,
                                   nccl_uid=None):
        """Create or retrieve an NCCL communicator for p2p tasks.

        Args:
            comm_key (str): communicator key.
            my_gpu_idx (int): the gpu index on the current process.
            peer_rank (int): the rank of the destination process.
            peer_gpu_idx (int): the gpu index on the peer process.
        Returns:
            communicator
        """
        # pylint: disable=unused-argument
        if not comm_key:
            raise RuntimeError("Got empty communicator key.")

        # TODO(Hao): lock the _dev_comm_map here.
        if comm_key in self._dev_comm_map:
            return self._dev_comm_map[comm_key]

        # Note (Hao): This is a bit complex so I decide to take a note here.
        # Here we need to consider three cases:
        # Case 1: src_rank != dst_rank, hence the send and recv happen on
        # different process (actors/tasks); each process makes independent
        # collective calls and manages corresponding communicators.
        # Case 2: src_rank == dst_rank, src_gpu_idx == dst_gpu_idx; for
        # this case, we simply throw a RuntimeError;
        # Case 3: src_rank == dst_rank, src_gpu_idx != dst_gpu_idx, which
        # means the send and recv will be called on the same process. We
        # DO NOT support this case for now. We need to properly scope:
        # (1) communicators creation, and
        # (2) send/recv calls
        # using groupStart(（ and groupEnd() calls to avoid deadlocks.
        if self.rank < peer_rank:
            my_p2p_rank = 0
        elif self.rank > peer_rank:
            my_p2p_rank = 1
        else:
            raise RuntimeError(
                "Send and recv happens on the same process! "
                "alpa.collective does not support this case as of now. "
                "Alternatively, consider doing GPU to GPU memcpy?")
        group_key = self._generate_group_key(comm_key)
        if my_p2p_rank == 0:
            if nccl_uid is None:
                nccl_uid = self._generate_nccl_uid(group_key)
        else:
            if nccl_uid is None:
                rendezvous = Rendezvous(group_key)
                rendezvous.meet(timeout_s=3000)
                nccl_uid = rendezvous.get_nccl_id()
                # Recycle the NCCLUniqueIDStore named actor *pro-activately* to
                # avoid named actor leak.
                if rendezvous.get_access_counter() == 2:
                    logger.debug(
                        "NCCLUniqueID has been broadcasted. The "
                        "NCCLUniqueIDStore will go out of context and be "
                        "destroyed.")
                    rendezvous.destroy_store()

        comms = xe.nccl_create_communicators(2, [my_p2p_rank], [my_gpu_idx],
                                             nccl_uid,
                                             ENV.NCCL_USE_MULTISTREAM.val)
        self._dev_comm_map[comm_key] = comms
        return comms

    def _generate_group_key(self, comm_key):
        """Generate a unique key used to initialize the KV store.

        The group key is a concatenation of the communicator key and
        the group name, following: [comm_key]@[group_name].
        """
        return comm_key + "@" + self.group_name

    @staticmethod
    def _destroy_store(group_key):
        """Destroy the KV store (Ray named actor).

        Args:
            group_key (str): the unique key to retrieve the KV store.

        Returns:
            None
        """
        store_name = get_store_name(group_key)
        try:
            store = ray.get_actor(store_name)
            ray.kill(store)
        except ValueError:
            logger.info(f"The store with name {store_name} has been destroyed "
                        f"somewhere else.")

    @staticmethod
    def generate_nccl_uid():
        group_uid = xla_nccl_util.get_nccl_unique_id()
        return group_uid

    @staticmethod
    def _generate_nccl_uid(key):
        """Generate an NCCL unique ID for initializing communicators.

        The method will also create a KV store using Ray named actor and store
        the NCCLUniqueID in the store. The store needs to be garbage collected
        when destroying the collective group.

        Args:
            key (str): the key for storage of NCCLUniqueID.

        Returns:
            NCCLUniqueID (str): NCCL unique ID.
        """
        group_uid = xla_nccl_util.get_nccl_unique_id()
        store_name = get_store_name(key)
        # Avoid a potential circular dependency in ray/actor.py
        from alpa.collective.util import NCCLUniqueIDStore  # pylint: disable=import-outside-toplevel
        store = NCCLUniqueIDStore.options(
            name=store_name, lifetime="detached").remote(store_name)
        ray.get([store.set_id.remote(group_uid)])
        return group_uid

    def create_p2p_communicator(self,
                                my_gpu_idx: int,
                                peer_rank: int,
                                peer_gpu_idx: int,
                                nccl_uid: str = None):
        """A public method to create p2p communicators

        Args:
            my_gpu_idx (int): the gpu index on self rank.
            peer_rank (int): the rank of the peer process.
            peer_gpu_idx (int): the index of the gpu on the peer process.
            nccl_uid (str, optional): optionally to provide the NCCLUniqueID in
                advance.

        Returns:
            None
        """
        comm_key = _get_comm_key_send_recv(self.rank, my_gpu_idx, peer_rank,
                                           peer_gpu_idx)
        self._get_nccl_p2p_communicator(comm_key, my_gpu_idx, peer_rank,
                                        peer_gpu_idx, nccl_uid)

    def allreduce(self, tensors, allreduce_options=AllReduceOptions()):
        raise NotImplementedError()

    def barrier(self, barrier_options=BarrierOptions()):
        raise NotImplementedError()

    def reduce(self, tensors, reduce_options=ReduceOptions()):
        raise NotImplementedError()

    def allgather(self,
                  tensor_lists,
                  tensors,
                  allgather_options=AllGatherOptions()):
        raise NotImplementedError()

    def broadcast(self, tensors, broadcast_options=BroadcastOptions()):
        raise NotImplementedError()

    def reducescatter(self,
                      tensors,
                      tensor_lists,
                      reducescatter_options=ReduceScatterOptions()):
        raise NotImplementedError()


def _get_comm_key_send_recv(my_rank, my_gpu_idx, peer_rank, peer_gpu_idx):
    """Return a key given source and destination ranks for p2p tasks.

    The p2p key is in the following form:
                [min_rank]_[gpu_index]:[max_rank]_[gpu_index].

    Args:
        my_rank (int): the rank of the source process.
        my_gpu_idx (int): the source gpu index on the process.
        peer_rank (int): the rank of the destination process.
        peer_gpu_idx (int): the destination gpu index on the process.

    Returns:
        comm_key (str): a string key to query the communication cache.
    """
    if my_rank < peer_rank:
        lower_key = str(my_rank) + "_" + str(my_gpu_idx)
        higher_key = str(peer_rank) + "_" + str(peer_gpu_idx)
    elif my_rank > peer_rank:
        lower_key = str(peer_rank) + "_" + str(peer_gpu_idx)
        higher_key = str(my_rank) + "_" + str(my_gpu_idx)
    else:
        raise RuntimeError(
            "Send and recv happens on the same process. alpa.collective "
            "does not support this case as of now. Alternatively, consider "
            "doing GPU to GPU memcpy?")
    comm_key = lower_key + ":" + higher_key
    return comm_key

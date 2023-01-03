"""NCCL-based collective operations."""
import logging

import ray
import cupy
from jax._src.lib import xla_bridge as xb, xla_extension as xe

from alpa.collective.const import ENV
from alpa.collective.collective_group import nccl_util
from alpa.collective.collective_group.base_collective_group import BaseGroup, Rendezvous
from alpa.collective.const import get_store_name
from alpa.collective.types import (AllReduceOptions, BarrierOptions, Backend,
                                   ReduceOptions, BroadcastOptions,
                                   AllGatherOptions, ReduceScatterOptions,
                                   SendOptions, RecvOptions)
from alpa.collective.collective_group.cuda_stream import get_stream_pool
from alpa.monkey_patch import override_get_backend

logger = logging.getLogger(__name__)


# FIXME: should not assume that each worker has the same number of devices
class NCCLGroup(BaseGroup):
    """NCCL-based collective operations."""

    def __init__(self, world_size, rank, group_name):
        """Init an NCCL collective group."""
        super().__init__(world_size, rank, group_name)

        # communicator and stream cache.
        # TODO (Hao): we need a lock here...
        self._barrier_tensor = None
        self._dev_comm_map = {}
        self._dev_streams_map = {}
        self._xla_comm_keys = set()

        # record the used GPU IDs.
        self._used_gpu_indices = set()

        # TODO(Fu): might need an event map
        self._dev_event_map = {}
        # This is only for cross-mesh all-reduce to use
        backend = override_get_backend()
        self.xla_comm_group = xe.CommGroup(backend)

        if nccl_util.get_nccl_build_version() < 2000:
            raise RuntimeError("NCCL in Ray requires NCCL >= 2.0.")
        if nccl_util.get_nccl_runtime_version() < 2704:
            logger.warning("NCCL send/recv calls requires NCCL>=2.7.4")

    def destroy_group(self):
        """Destroy the group and release NCCL communicators."""
        if len(self._dev_comm_map.keys()) > 0:

            # TODO(Hao): check this barrier call
            # self.barrier()

            # Destroy the communicators and streams.
            for comm_key, comms in self._dev_comm_map.items():
                for c in comms:
                    # FIXME(yonghao): comms created in XLA should be destroied
                    if hasattr(c, "destroy"):
                        c.destroy()
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

    def allreduce(self, tensors, allreduce_options=AllReduceOptions()):
        """AllReduce tensors across the collective group following options.

        Args:
            tensors (List): the list of tensors to be reduced. Each tensor must
                            reside on one GPU of the current process.
            allreduce_options: allreduce options.

        Returns:
            None
        """

        def collective_fn(input_tensor, output_tensor, comm, stream):
            comm.allReduce(
                nccl_util.get_tensor_ptr(input_tensor),
                nccl_util.get_tensor_ptr(output_tensor),
                nccl_util.get_tensor_n_elements(input_tensor),
                nccl_util.get_nccl_tensor_dtype(input_tensor),
                nccl_util.get_nccl_reduce_op(allreduce_options.reduce_op),
                stream.ptr)

        self._collective(tensors, tensors, collective_fn)

    def barrier(self, barrier_options=BarrierOptions()):
        """Blocks until all processes reach this barrier.

        Args:
            barrier_options: barrier options.

        Returns:
            None
        """
        # Get the device list.
        if self._used_gpu_indices:
            devices = list(self._used_gpu_indices)
        else:
            devices = list(range(nccl_util.get_num_gpus()))
        barrier_tensors = [None] * len(devices)
        for i, d in enumerate(devices):
            with nccl_util.Device(d):
                barrier_tensors[i] = cupy.array([1])
        self.allreduce(barrier_tensors)

    def reduce(self, tensors, reduce_options=ReduceOptions()):
        """Reduce tensors to a destination gpu following options.

        Args:
            tensors (List): the list of tensors to be reduced, each tensor
                            must reside on one gpu of the current process.
            reduce_options: reduce options.

        Returns:
            None
        """
        root_rank = (len(tensors) * reduce_options.root_rank +
                     reduce_options.root_tensor)

        def collective_fn(input_tensor, output_tensor, comm, stream):
            comm.reduce(nccl_util.get_tensor_ptr(input_tensor),
                        nccl_util.get_tensor_ptr(output_tensor),
                        nccl_util.get_tensor_n_elements(input_tensor),
                        nccl_util.get_nccl_tensor_dtype(input_tensor),
                        nccl_util.get_nccl_reduce_op(reduce_options.reduce_op),
                        root_rank, stream.ptr)

        self._collective(tensors, tensors, collective_fn)

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

        def collective_fn(input_tensor, output_tensor, comm, stream):
            comm.broadcast(
                nccl_util.get_tensor_ptr(input_tensor),
                nccl_util.get_tensor_ptr(output_tensor),
                broadcast_options.n_elements if broadcast_options.n_elements > 0
                else nccl_util.get_tensor_n_elements(input_tensor),
                nccl_util.get_nccl_tensor_dtype(input_tensor), root_rank,
                stream.ptr)

        _check_gpu_tensors(tensors)

        key = broadcast_options.comm_key
        comms = self._get_nccl_broadcast_communicator(
            key, broadcast_options.world_size, broadcast_options.devices_ids,
            broadcast_options.devices_global_rank)
        streams = self._dev_streams_map[key]
        events = self._dev_event_map[key]
        self._sync_streams(broadcast_options.devices_ids, events, streams)

        nccl_util.groupStart()
        for i, tensor in enumerate(tensors):
            collective_fn(tensor, tensor, comms[i], streams[i])
        nccl_util.groupEnd()

    def _get_nccl_broadcast_communicator(self,
                                         comm_key,
                                         world_size,
                                         devices_ids,
                                         devices_global_rank,
                                         nccl_uid=None):
        """Create or retrieve an NCCL communicator for broadcast from cache.
        Here we only use partial devices in a host, so we create this function
        besides _get_nccl_collective_communicator.

        If the communicator is found in cache, return the communicator. If not,
        a communicator and a stream will be created and put in cache.

        Args:
            comm_key (str): the key to query the communicator cache.
            world_size (int): the number of devices in this collective
                              communicator.
            devices_ids (List): a list of GPU devices of the current process
                                that participates into the collective.
            devices_global_rank (List): the corresponding global rank for device
                                        in devices_ids.
            nccl_uid : If it is None, we will create a nccl_uid here.

        Returns:
            communicator: the NCCL communicator corresponded to the devices.
        """
        if not comm_key:
            raise RuntimeError("Got empty communicator key.")

        # TODO(Hao): lock the _dev_comm_map here.
        if comm_key in self._dev_comm_map:
            return self._dev_comm_map[comm_key]

        for d in devices_ids:
            self._used_gpu_indices.add(d)

        nccl_uid = self._rendezvous_nccl_uid(devices_global_rank[0], comm_key,
                                             self.world_size, nccl_uid)

        # Now create the communicators
        comms = [None] * len(devices_ids)
        streams = [None] * len(devices_ids)
        events = [None] * len(devices_ids)
        nccl_util.groupStart()
        for i, (global_rank,
                device_id) in enumerate(zip(devices_global_rank, devices_ids)):
            with nccl_util.Device(device_id):
                comms[i] = nccl_util.create_nccl_communicator(
                    world_size, nccl_uid, global_rank)
                streams[i] = get_stream_pool(device_id).get_stream()
                events[i] = cupy.cuda.Event()
        nccl_util.groupEnd()
        self._dev_comm_map[comm_key] = comms
        self._dev_streams_map[comm_key] = streams
        self._dev_event_map[comm_key] = events
        return comms

    def broadcast(self, tensors, broadcast_options=BroadcastOptions()):
        """Broadcast tensors to all other gpus following options.

        Args:
            tensors (List): tensors to be broadcast or received.
            broadcast_options: broadcast options.

        Returns:
            None
        """
        root_rank = (len(tensors) * broadcast_options.root_rank +
                     broadcast_options.root_tensor)

        def collective_fn(input_tensor, output_tensor, comm, stream):
            comm.broadcast(nccl_util.get_tensor_ptr(input_tensor),
                           nccl_util.get_tensor_ptr(output_tensor),
                           nccl_util.get_tensor_n_elements(input_tensor),
                           nccl_util.get_nccl_tensor_dtype(input_tensor),
                           root_rank, stream.ptr)

        self._collective(tensors, tensors, collective_fn)

    def allgather(self,
                  tensor_lists,
                  tensors,
                  allgather_options=AllGatherOptions()):
        """Allgather tensors across gpus into a list of tensors.

        Args:
            tensor_lists (List[List[Tensor]]): allgathered tensors.
            tensors: the list of tensors to allgather across the group.
                     Each tensor must lolcate on a GPU of the process.
            allgather_options: allgather options.

        Returns:
            None
        """

        def collective_fn(input_tensor, output_tensor, comm, stream):
            comm.allGather(nccl_util.get_tensor_ptr(input_tensor),
                           nccl_util.get_tensor_ptr(output_tensor),
                           nccl_util.get_tensor_n_elements(input_tensor),
                           nccl_util.get_nccl_tensor_dtype(input_tensor),
                           stream.ptr)

        _check_inputs_compatibility_for_scatter_gather(tensors, tensor_lists)
        output_flattened = [
            _flatten_for_scatter_gather(tensor_list, copy=False)
            for tensor_list in tensor_lists
        ]

        def postprocess_fn(stream):
            # pylint: disable=unused-argument
            # TODO(Hao): designate a copy stream.
            for i, tensor_list in enumerate(tensor_lists):
                for j, tensor in enumerate(tensor_list):
                    nccl_util.copy_tensor(tensor, output_flattened[i][j])

        self._collective(tensors,
                         output_flattened,
                         collective_fn,
                         postprocess_fn=postprocess_fn)

    def reducescatter(self,
                      tensors,
                      tensor_lists,
                      reducescatter_options=ReduceScatterOptions()):
        """Reduce then scatter a list of tensors across the group.

        Args:
            tensors (List): the output tensors (could be unspecified), each
                            located on a GPU of the current process.
            tensor_lists (List[List]): the list of tensors to be reduced then
                                       scattered.
            reducescatter_options: reduce-scatter options.

        Returns:
            None
        """

        def collective_fn(input_tensor, output_tensor, comm, stream):
            comm.reduceScatter(
                nccl_util.get_tensor_ptr(input_tensor),
                nccl_util.get_tensor_ptr(output_tensor),
                nccl_util.get_tensor_n_elements(output_tensor),
                nccl_util.get_nccl_tensor_dtype(output_tensor),
                nccl_util.get_nccl_reduce_op(reducescatter_options.reduce_op),
                stream.ptr)

        _check_inputs_compatibility_for_scatter_gather(tensors, tensor_lists)
        input_flattened = [
            _flatten_for_scatter_gather(tensor_list, copy=False)
            for tensor_list in tensor_lists
        ]

        def preprocess_fn(stream):
            # pylint: disable=unused-argument
            for i, tensor_list in enumerate(tensor_lists):
                for j, tensor in enumerate(tensor_list):
                    nccl_util.copy_tensor(input_flattened[i][j], tensor)

        self._collective(input_flattened,
                         tensors,
                         collective_fn,
                         preprocess_fn=preprocess_fn)

    def send(self, tensors, send_options=SendOptions()):
        """Send a tensor to a destination gpu in the group.

        Args:
            tensors (List): the tensor to send.
            send_options: send options.

        Returns:
            None
        """

        def p2p_fn(tensor, comm, stream, peer):
            comm.send(
                nccl_util.get_tensor_ptr(tensor),
                send_options.n_elements if send_options.n_elements > 0 else
                nccl_util.get_tensor_n_elements(tensor),
                nccl_util.get_nccl_tensor_dtype(tensor), peer, stream.ptr)

        self._point2point(tensors, p2p_fn, send_options.dst_rank,
                          send_options.dst_gpu_index)

    def recv(self, tensors, recv_options=RecvOptions()):
        """Receive a tensor from a source gpu in the group.

        Args:
            tensors (List): the received tensor.
            recv_options: Receive options.

        Returns:
            None
        """

        def p2p_fn(tensor, comm, stream, peer):
            comm.recv(
                nccl_util.get_tensor_ptr(tensor),
                recv_options.n_elements if recv_options.n_elements > 0 else
                nccl_util.get_tensor_n_elements(tensor),
                nccl_util.get_nccl_tensor_dtype(tensor), peer, stream.ptr)

        self._point2point(tensors, p2p_fn, recv_options.src_rank,
                          recv_options.src_gpu_index)

    def _get_nccl_collective_communicator(self, comm_key, device_list):
        """Create or retrieve an NCCL communicator from cache.

        If the communicator is found in cache, return the communicator. If not,
        a communicator and a stream will be created and put in cache.
        TODO(Hao): this function is not thread-safe now.

        Args:
            comm_key (str): the key to query the communicator cache.
            device_list (List): a list of GPU devices of the current process
                                that participates into the collective.

        Returns:
            communicator: the NCCL communicator corresponded to the devices.
        """
        if not comm_key:
            raise RuntimeError("Got empty communicator key.")

        # TODO(Hao): lock the _dev_comm_map here.
        if comm_key in self._dev_comm_map:
            return self._dev_comm_map[comm_key]

        for d in device_list:
            self._used_gpu_indices.add(d)

        nccl_uid = self._rendezvous_nccl_uid(self.rank, comm_key,
                                             self.world_size)

        # Now create the communicators
        actual_world_size = len(device_list) * self.world_size
        comms = [None] * len(device_list)
        streams = [None] * len(device_list)
        events = [None] * len(device_list)

        nccl_util.groupStart()
        for i, device in enumerate(device_list):
            actual_rank = self.rank * len(device_list) + i
            with nccl_util.Device(device):
                comms[i] = nccl_util.create_nccl_communicator(
                    actual_world_size, nccl_uid, actual_rank)
                # request a stream from the pool
                # note the device_idx is absolute index.
                streams[i] = get_stream_pool(device).get_stream()
                # TODO(Fu): double check the parameters
                events[i] = cupy.cuda.Event()
        nccl_util.groupEnd()
        # TODO(Fu): lock
        self._dev_comm_map[comm_key] = comms
        self._dev_streams_map[comm_key] = streams
        self._dev_event_map[comm_key] = events
        return comms

    def create_nccl_collective_communicator(self, devices):
        key = _get_comm_key_from_devices(devices)
        self._get_nccl_collective_communicator(key, devices)

    def create_and_set_xla_communicators(self, devices, key):
        comm_key = _get_comm_key_from_devices(devices)
        if comm_key in self._xla_comm_keys:
            return
        for d in devices:
            self._used_gpu_indices.add(d)

        nccl_uid = self._rendezvous_nccl_uid(self.rank, comm_key,
                                             self.world_size)

        # Now create the communicators
        actual_world_size = len(devices) * self.world_size
        # FIXME: pass the start rank at the initial point
        start_rank = self.rank * len(devices)
        actual_ranks = [start_rank + i for i in range(len(devices))]
        local_ids = list(range(len(devices)))
        self.xla_comm_group.nccl_create_communicators(actual_world_size,
                                                      actual_ranks, local_ids,
                                                      nccl_uid)

        xe.set_comm_group_info(key, self.xla_comm_group, nccl_uid)
        self._xla_comm_keys.add(comm_key)

    @staticmethod
    def _sync_streams(device_list, events, streams):
        """Let NCCL streams wait for current streams for every device."""
        # TODO(Fu): recordStream besides calling this function?
        if ENV.NCCL_USE_MULTISTREAM.val:
            for i, device in enumerate(device_list):
                with nccl_util.Device(device):
                    events[i].record(cupy.cuda.get_current_stream())
                    streams[i].wait_event(events[i])

    def _get_nccl_p2p_communicator(self,
                                   comm_key,
                                   my_gpu_idx,
                                   peer_rank,
                                   peer_gpu_idx,
                                   nccl_uid=None):
        """Create or retrieve an NCCL communicator for p2p tasks.

        Note(Hao): this function is not thread-safe now.

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
        nccl_uid = self._rendezvous_nccl_uid(my_p2p_rank, comm_key, 2, nccl_uid)

        # create the p2p communicators
        with nccl_util.Device(my_gpu_idx):
            comm = nccl_util.create_nccl_communicator(2, nccl_uid, my_p2p_rank)
            stream = get_stream_pool(my_gpu_idx).get_stream()
            event = cupy.cuda.Event()

        self._dev_comm_map[comm_key] = [comm]
        self._dev_streams_map[comm_key] = [stream]
        self._dev_event_map[comm_key] = [event]
        return [comm]

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
        group_uid = nccl_util.get_nccl_unique_id()
        return group_uid

    @staticmethod
    def _generate_nccl_uid(key):
        """Generate an NCCL unique ID for initializing communicators.

        The method will also create a KV store using Ray named actor and store
        the NCCLUniqueID in the store. The store needs to be garbage collected
        when destroying the collective group.

        Args:
            key (str): the key of the .

        Returns:
            NCCLUniqueID (str): NCCL unique ID.
        """
        group_uid = nccl_util.get_nccl_unique_id()
        store_name = get_store_name(key)
        # Avoid a potential circular dependency in ray/actor.py
        from alpa.collective.util import NCCLUniqueIDStore  # pylint: disable=import-outside-toplevel
        store = NCCLUniqueIDStore.options(
            name=store_name, lifetime="detached").remote(store_name)
        ray.get([store.set_id.remote(group_uid)])
        return group_uid

    def _collective(self,
                    input_tensors,
                    output_tensors,
                    collective_fn,
                    preprocess_fn=None,
                    postprocess_fn=None):
        """A method to encapsulate all collective calls.

        Args:
            input_tensors: the list of the input tensors.
            output_tensors: the list of the output tensors.
            collective_fn: the collective function call.
            preprocess_fn: preprocess procedures before collective calls.
            postprocess_fn: postprocess procedures after collective calls.

        Returns:
            None
        """
        _check_gpu_tensors(input_tensors)
        _check_gpu_tensors(output_tensors)

        devices = nccl_util.get_tensor_device_list(input_tensors)
        key = _get_comm_key_from_devices(devices)
        comms = self._get_nccl_collective_communicator(key, devices)
        streams = self._dev_streams_map[key]
        events = self._dev_event_map[key]

        # TODO(Hao): sync streams and events
        self._sync_streams(devices, events, streams)

        # Make the collective call
        if preprocess_fn:
            preprocess_fn(streams)

        nccl_util.groupStart()
        # TODO(Fu): how to recordStreams as there are no library functions
        # We also need to make sure input tensors are not freed before their
        # usages on ncclStreams finish. This can be achieved by calling
        # c10::cuda::CUDACachingAllocator::recordStream, which remembers the
        # usage stream (ncclStream), creates an event on the usage stream
        # when GC attempts to free the input tensor, and delays GC until that
        # event is done.
        for i, tensor in enumerate(input_tensors):
            collective_fn(tensor, output_tensors[i], comms[i], streams[i])
        nccl_util.groupEnd()
        if postprocess_fn:
            postprocess_fn(streams)

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

    def create_nccl_broadcast_communicator(self,
                                           comm_key,
                                           world_size,
                                           devices_ids,
                                           devices_global_rank,
                                           nccl_uid=None):
        self._get_nccl_broadcast_communicator(comm_key, world_size, devices_ids,
                                              devices_global_rank, nccl_uid)

    def _point2point(self, tensors, p2p_fn, peer_rank: int, peer_gpu_idx: int):
        """A method to encapsulate all peer-to-peer calls (i.e., send/recv).

        Args:
            tensors: the tensor to send or receive.
            p2p_fn: the p2p function call.
            peer_rank (int): the rank of the peer process.
            peer_gpu_idx (int): the index of the gpu on the peer process.

        Returns:
            None
        """
        # check send/recv availability.
        if nccl_util.get_nccl_runtime_version() < 2704:
            raise RuntimeError("P2p send/recv requires NCCL >= 2.7.4. "
                               f"Got '{nccl_util.get_nccl_runtime_version()}'.")
        _check_gpu_tensors(tensors)

        # we currently only support single device to single device send/recv.
        assert len(tensors) == 1
        my_gpu_idx = nccl_util.get_tensor_device(tensors[0])
        comm_key = _get_comm_key_send_recv(self.rank, my_gpu_idx, peer_rank,
                                           peer_gpu_idx)
        comms = self._get_nccl_p2p_communicator(comm_key, my_gpu_idx, peer_rank,
                                                peer_gpu_idx)
        streams = self._dev_streams_map[comm_key]
        events = self._dev_event_map[comm_key]

        # TODO(Hao): sync streams and events
        self._sync_streams([my_gpu_idx], events, streams)

        # We have made sure that self.rank != peer_rank during API check.
        peer_p2p_rank = 0 if self.rank > peer_rank else 1
        for i, t in enumerate(tensors):
            p2p_fn(t, comms[i], streams[i], peer_p2p_rank)

    def _rendezvous_nccl_uid(self, rank, comm_key, max_counter, nccl_uid=None):
        group_key = self._generate_group_key(comm_key)
        if rank == 0:
            if nccl_uid is None:
                nccl_uid = self._generate_nccl_uid(group_key)
        else:
            if nccl_uid is None:
                rendezvous = Rendezvous(group_key)
                rendezvous.meet()
                nccl_uid = rendezvous.get_nccl_id()

                # Recycle the NCCLUniqueIDStore named actor *pro-activately* to
                # avoid named actor leak.
                if rendezvous.get_access_counter() == max_counter:
                    logger.debug(
                        "NCCLUniqueID has been broadcasted. The "
                        "NCCLUniqueIDStore will go out of context and be "
                        "destroyed.")
                    rendezvous.destroy_store()
        return nccl_uid


def _flatten_for_scatter_gather(tensor_list, copy=False):
    """Flatten the tensor for gather/scatter operations.

    Args:
        tensor_list: the list of tensors to be scattered/gathered.
        copy: whether the copy the tensors in tensor_list into the buffer.

    Returns:
        The flattened tensor buffer.
    """
    if not tensor_list:
        raise RuntimeError("Received an empty list.")
    t = tensor_list[0]
    # note we need a cupy dtype here.
    dtype = nccl_util.get_cupy_tensor_dtype(t)
    buffer_shape = [len(tensor_list)] + nccl_util.get_tensor_shape(t)
    device = nccl_util.get_tensor_device(t)
    with nccl_util.Device(device):
        buffer = cupy.empty(buffer_shape, dtype=dtype)
    if copy:
        for i, tensor in enumerate(tensor_list):
            nccl_util.copy_tensor(buffer[i], tensor)
    return buffer


def _check_inputs_compatibility_for_scatter_gather(tensors, tensor_lists):
    """Check the compatibility between tensor input and tensor list input."""
    if not tensors or not isinstance(tensors, list):
        raise RuntimeError(
            "The first argument 'tensors' expects a list of tensors.")
    if not tensor_lists or not isinstance(tensor_lists, list):
        raise RuntimeError("The second argument 'tensor_lists' "
                           "expects a list of tensor list.")
    dtype = nccl_util.get_nccl_tensor_dtype(tensors[0])
    shape = nccl_util.get_tensor_shape(tensors[0])
    for i, tl in enumerate(tensor_lists):
        # check all tensor in `tensors` match.
        dt = nccl_util.get_nccl_tensor_dtype(tensors[i])
        if dt != dtype:
            raise RuntimeError(
                "All tensor operands to scatter/gather must "
                f"have the same dtype. Got '{dt}' and '{dtype}'.")
        # Note: typically CCL libraries only requires they have the same
        # number of elements; Here we make it more strict -- we require
        # exact shape match.
        s = nccl_util.get_tensor_shape(tensors[i])
        if s != shape:
            raise RuntimeError("All tensor operands to scatter/gather must "
                               f"have the same shape. Got '{s}' and '{shape}'.")
        # check all tensors in `tensor_lists` match.
        for t in tl:
            # check dtype
            dt = nccl_util.get_nccl_tensor_dtype(t)
            if dt != dtype:
                raise RuntimeError(
                    "All tensor operands to scatter/gather must "
                    f"have the same dtype. Got '{dt}' and '{dtype}'.")
            s = nccl_util.get_tensor_shape(t)
            if s != shape:
                raise RuntimeError(
                    "All tensor operands to scatter/gather must "
                    f"have the same shape. Got '{s}' and '{shape}'.")


def _check_gpu_tensors(tensors):
    """Check all tensors are distributed on different GPUs."""
    if not tensors or not isinstance(tensors, list):
        raise RuntimeError("'tensors' must be a nonempty list.")
    if len(tensors) > nccl_util.get_num_gpus():
        raise RuntimeError("Tensor list cannot be larger than the number"
                           f"of available GPUs. Got {len(tensors)} > "
                           f"{nccl_util.get_num_gpus()}.")
    t0 = tensors[0]
    dt = nccl_util.get_nccl_tensor_dtype(t0)
    s = nccl_util.get_tensor_shape(t0)
    d = nccl_util.get_tensor_device(t0)
    for i, t in enumerate(tensors):
        if i == 0:
            continue
        # We need to check the following:
        # (1) tensor is cuda (already checked during API)
        # (2) tensor dtype
        # (3) tensor shape match
        # (4) each tensor is on a different GPU
        dtype = nccl_util.get_nccl_tensor_dtype(t)
        if dt != dtype:
            raise RuntimeError(
                f"Tensors must have identical dtypes. Got: '{dtype}'.")
        shape = nccl_util.get_tensor_shape(t)
        if s != shape:
            raise RuntimeError(
                f"Tensors must have identical shapes. Got: '{shape}'.")
        device = nccl_util.get_tensor_device(t)
        if device == d:
            raise RuntimeError("Tensor must be on distinct GPUs.")


def _get_comm_key_from_devices(devices):
    """Return a key from a list of devices for collective calls.

    For example, if the tensors are on gpus 0, 1, 2, 3,
    then the key would be "0,1,2,3".

    Args:
        devices(list): a list of GPU device indices

    Returns:
        str: a string represents the key to query the communicator cache.

    """
    return ",".join([str(d) for d in devices])


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

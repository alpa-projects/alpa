"""Cluster related configurations (e.g., topology)."""
from collections.abc import Iterable
import os
from operator import attrgetter
import time

import numpy as np
import ray
from jax import core, pmap, tree_util, xla
from jax import numpy as jnp
from jax.abstract_arrays import array_types
from jax.interpreters import pxla
from jax.interpreters.pxla import (ShardingSpec, Chunked, NoSharding, Replicated,
    ShardedAxis, _as_slice_indices, _hashable_index, ShardedDeviceArray)
from jax.lib import xla_client, xla_bridge
from jax._src.util import (partial, unzip3, prod, safe_map, safe_zip,
                           extend_name_stack, wrap_name, assert_unreachable,
                           tuple_insert, tuple_delete, curry)

from parax.util import get_dim_last_value, to_int_tuple
from parax.xla_pass_context import XlaPassContext


class LogicalDeviceMesh:
    """
    A logical multi-dimensional device mesh.

    Each mesh dimension has its own latency and bandwidth.
    We use alpha-beta model to model the communication cost.
    """

    def __init__(self, physical_mesh, id_mesh, mesh_alpha=None, mesh_beta=None):
        self.physical_mesh = physical_mesh
        self.id_mesh = np.array(id_mesh)
        self.flatten_ids = tuple(int(x) for x in self.id_mesh.flatten())
        self.is_multi_host = False

        # coefficient for alpha-beta communication model
        if mesh_alpha is None:
            mesh_alpha = [1] * len(self.id_mesh.shape)
        if mesh_beta is None:
            mesh_beta = [1] * len(self.id_mesh.shape)
        self.mesh_alpha = tuple(mesh_alpha)
        self.mesh_beta = tuple(mesh_beta)

    def all_gather_cost(self, num_bytes, mesh_dim):
        num_devices = self.id_mesh.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] +
                self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes +
                0.1)

    def all_reduce_cost(self, num_bytes, mesh_dim):
        num_devices = self.id_mesh.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] +
                self.mesh_beta[mesh_dim] * 2 * (num_devices - 1) / num_devices * num_bytes +
                0.01)

    def reduce_scatter_cost(self, num_bytes, mesh_dim):
        num_devices = self.id_mesh.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] +
                self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes +
                0.001)

    def get_tensor_dim_to_mesh_dim(self, shape,
                                   tile_assignment_dimensions, tile_assignment_devices):
        tile_assignment = np.array(tile_assignment_devices).reshape(tile_assignment_dimensions)

        tensor_dim_vals = tuple(get_dim_last_value(tile_assignment, i)
            for i in range(len(shape)))

        mesh_dim_vals = tuple(get_dim_last_value(self.id_mesh, j)
            for j in range(len(self.id_mesh.shape)))

        ret = [-1] * len(shape)
        for i in range(len(shape)):
            if tile_assignment_dimensions[i] != 1:
                found = False
                for j in range(len(self.id_mesh.shape)):
                    if tensor_dim_vals[i] == mesh_dim_vals[j]:
                        ret[i] = j
                        found = True
                assert found

        return ret

    def make_replicated_spec(self, array):
        sharding = (NoSharding(),) * len(array.shape)
        mesh_mapping = (Replicated(len(self.flatten_ids)),)
        return ShardingSpec(sharding, mesh_mapping)

    def make_tile_spec(self, array, tensor_dims, mesh_dims):
        shape = array.shape
        sharding = [NoSharding(),] * len(shape)
        mesh_mapping = [None,] * len(self.id_mesh.shape)

        for i, (tensor_dim, mesh_dim) in enumerate(zip(tensor_dims, mesh_dims)):
            sharding[tensor_dim] = Chunked([self.id_mesh.shape[mesh_dim]],)
            mesh_mapping[mesh_dim] = ShardedAxis(i)

        for i in range(len(mesh_mapping)):
            if mesh_mapping[i] is None:
                mesh_mapping[i] = Replicated(self.id_mesh.shape[i])

        return ShardingSpec(sharding, mesh_mapping)

    def __hash__(self):
        return hash((self.flatten_ids, self.id_mesh.shape,
                     self.mesh_alpha, self.mesh_beta))

    def __eq__(self, other):
        return (self.flatten_ids, self.id_mesh.shape,
                self.mesh_alpha, self.mesh_beta) ==\
               (other.flatten_ids, other.id_mesh.shape,
                other.mesh_alpha, other.mesh_beta)


class SingleHostDeviceMesh:
    """A physical device mesh that presents devices on a single node."""

    def __init__(self, devices):
        self.devices = devices

    def get_logical_mesh(self, mesh_shape, mesh_alpha=None, mesh_beta=None):
        """Get a mapping to logoical mesh."""
        device_ids = np.array([d.id for d in self.devices])
        device_ids = device_ids.reshape(mesh_shape)
        mesh_alpha = mesh_alpha or (1.0,) * len(mesh_shape)
        mesh_beta = mesh_beta or (1.0,) * len(mesh_shape)
        return LogicalDeviceMesh(self, device_ids, mesh_alpha, mesh_beta)

    def get_default_logical_mesh(self):
        return self.get_logical_mesh((1, len(self.devices)))

    def get_callable_with_arg_handler(self, compiled, avals, out_avals,
            input_sharding_specs, output_sharding_specs):
        input_indices = [pxla.spec_to_indices(aval.shape, spec) for
                         aval, spec in zip(avals, input_sharding_specs)]
        args_handler = partial(pxla.shard_args, self.devices, input_indices)
        outs_handler = pxla.avals_to_results_handler(1, len(self.devices),
                                                     output_sharding_specs, out_avals)

        ret = partial(SingleHostDeviceMesh._execute_with_handler,
            compiled, args_handler, outs_handler)
        ret.shard_args_only = partial(self.preshard_args, args_handler, avals,
            input_sharding_specs, input_indices)
        return ret

    def preshard_args(self, handler, avals, sharding_specs, indices, *args):
        input_bufs = handler(args)

        sharded_args = []
        for i in range(len(args)):
            sharded_args.append(ShardedDeviceArray(
                avals[i],
                sharding_specs[i],
                input_bufs[i],
                indices[i],
            ))

        return sharded_args

    @staticmethod
    def _execute_with_handler(compiled, args_handler, outs_handler, *args):
        input_bufs = args_handler(args)
        out_bufs = compiled.execute_sharded_on_local_devices(input_bufs)
        return outs_handler(out_bufs)


class RemoteExecutableRef:
    """A reference to a remote compiled XLA executable binary"""

    ct = 0

    def __init__(self, device_mesh):
        self.device_mesh = device_mesh
        self.uuid = RemoteExecutableRef.ct
        RemoteExecutableRef.ct = (RemoteExecutableRef.ct + 1) % (1 << 60)

    def __repr__(self):
        return f"RemoteExecutableRef(uuid = {self.uuid})"

    def __del__(self):
        self.device_mesh.delete_remote_executable(self)


class RemoteBufferRef:
    """A refernece to a remote device buffer."""

    ct = 0

    def __init__(self, device_mesh, host_id, device_id):
        self.device_mesh = device_mesh
        self.host_id = host_id
        self.device_id = device_id
        self.uuid = RemoteBufferRef.ct
        RemoteBufferRef.ct = (RemoteBufferRef.ct + 1) % (1 << 60)

    def __repr__(self):
        return f"RemoteBufferRef(uuid = {self.uuid}, loc = ({self.host_id}, {self.device_id}))"

    def __del__(self):
        self.device_mesh.delete_remote_buffers((self,))


class DistributedArray:
    """A distributed array on a MultiHostDeviceMesh."""

    def __init__(self, device_mesh, aval, sharding_spec, remote_buffers, indices):
        self.device_mesh = device_mesh
        self.aval = aval
        self.sharding_spec = sharding_spec
        self.remote_buffers = remote_buffers
        self.indices = indices

        self._npy_value = None
        self._one_replica_buffer_indices = None

    @property
    def one_replica_buffer_indices(self):
        """Indices of buffers containing one complete copy of the array data."""
        if self._one_replica_buffer_indices is None:
            one_replica_indices = []
            seen_index_hashes = set()
            for i, index in enumerate(self.indices):
                hashed_index = _hashable_index(index)
                if hashed_index not in seen_index_hashes:
                    one_replica_indices.append(i)
                    seen_index_hashes.add(hashed_index)
            self._one_replica_buffer_indices = one_replica_indices
        return self._one_replica_buffer_indices

    @property
    def _value(self):
        if self._npy_value is None:
            npy_value = np.empty(self.aval.shape, self.aval.dtype)
            fetched_np_buffers = self.device_mesh.get_remote_buffers([
                self.remote_buffers[i] for i in self.one_replica_buffer_indices
            ])
            for i in self.one_replica_buffer_indices:
                npy_value[self.indices[i]] = fetched_np_buffers[i]
            self._npy_value = npy_value
        return self._npy_value

    def __array__(self, dtype=None, context=None):
        return np.asarray(self._value, dtype=dtype)

    def __str__(self):
        return str(self._value)


core.pytype_aval_mappings[DistributedArray] = attrgetter('aval')
xla.pytype_aval_mappings[DistributedArray] = attrgetter('aval')
xla.canonicalize_dtype_handlers[DistributedArray] = lambda x: x


def get_uuid_np_array(array):
    """Convert a np array of RemoteBufferRef to a np array of UUID (int64)."""
    ret = np.empty(array.shape, dtype=np.int64)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ret[i][j] = array[i][j].uuid
    return ret


class MultiHostDeviceMesh:
    """A physical device mesh that presents a device mesh on multipe nodes."""

    def __init__(self, host_ids, host_info, num_devices_per_host, head_ip):
        self.host_ids = host_ids
        self.host_info = host_info
        self.num_hosts = len(self.host_ids)
        self.num_devices_per_host = num_devices_per_host
        self.head_ip = head_ip
        self.total_devices = self.num_hosts * self.num_devices_per_host

        # Launch distributed xla runtime
        port = np.random.randint(10000, 11000)
        self.server_address = f"{self.head_ip}:{port}"
        self.service_server = None
        self.service_server = xla_client._xla.get_distributed_runtime_service(
            self.server_address, self.num_hosts)
        time.sleep(0.5)

        # Launch workers
        self.workers = []
        for i in range(self.num_hosts):
            node_resource = "node:" + self.host_info[i]["NodeManagerAddress"]
            cls = ray.remote(num_gpus=self.num_devices_per_host,
                             resources={node_resource: 1e-3})(MeshHostWorker)
            self.workers.append(cls.remote(self.server_address, i))
        self.sync_workers()

    def get_logical_mesh(self, mesh_shape, mesh_alpha=None, mesh_beta=None):
        """Get a mapping to logoical mesh."""
        id_mesh = np.arange(self.num_hosts * self.num_devices_per_host).\
            reshape(mesh_shape)
        mesh_alpha = mesh_alpha or (1.0,) * len(mesh_shape)
        mesh_beta = mesh_beta or (1.0,) * len(mesh_shape)
        return LogicalDeviceMesh(self, id_mesh, mesh_alpha, mesh_beta)

    def get_default_logical_mesh(self):
        return self.get_logical_mesh((self.num_hosts, self.num_devices_per_host),
            [1, 1], [1, 0.01])

    def compile_remote_executable(self,
                                  hlo_proto,
                                  logical_mesh_shape,
                                  auto_sharding_strategy_vector,
                                  is_tuple_args):
        executable = RemoteExecutableRef(self)
        for w in self.workers:
            w.compile_executable.remote(
                executable.uuid,
                hlo_proto,
                logical_mesh_shape,
                auto_sharding_strategy_vector,
                is_tuple_args)
        return executable

    def get_remote_buffers(self, buf_refs):
        obj_refs = []
        for buf_ref in buf_refs:
            obj_refs.append(self.workers[buf_ref.host_id].
                get_buffers.remote(buf_ref.uuid))

        return ray.get(obj_refs)

    def delete_remote_buffers(self, buf_refs):
        if self.workers is None:
            return

        for buf_ref in buf_refs:
            self.workers[buf_ref.host_id].delete_buffers.remote(buf_ref.uuid)

    def delete_remote_executable(self, exe_ref):
        if self.workers is None:
            return

        for i in range(self.num_hosts):
            self.workers[i].delete_executable.remote(exe_ref.uuid)

    def sync_workers(self):
        ray.get([w.sync.remote() for w in self.workers])

    def shutdown_workers(self):
        ray.get([w.shutdown.remote() for w in self.workers])

    def get_callable_with_arg_handler(self, remote_executable, avals, out_avals,
            input_sharding_specs, output_sharding_specs):
        input_indices = [pxla.spec_to_indices(aval.shape, spec) for
                         aval, spec in zip(avals, input_sharding_specs)]
        args_handler = partial(self._shard_args, input_indices)

        output_indices = [pxla.spec_to_indices(aval.shape, spec) for
                          aval, spec in zip(out_avals, output_sharding_specs)]

        outs_handler = partial(self._gather_outs, out_avals, output_sharding_specs, output_indices)
        return partial(self._execute_with_handler,
            remote_executable, args_handler, outs_handler, len(out_avals))

    def _shard_args(self, arg_indices, args):
        input_bufs = []
        for arg, indices in zip(args, arg_indices):
            # Fast path for DistributedArray
            if isinstance(arg, DistributedArray) and arg.indices == indices:
                return arg.remote_buffers
            else:  # Slow path
                arg = xla.canonicalize_dtype(arg)
                buf_refs = shard_arg_handlers[type(arg)](arg, self, indices)
                input_bufs.append(buf_refs)

        return input_bufs

    def _gather_outs(self, avals, sharding_specs, indices, bufs):
        ret = []
        for i in range(len(avals)):
            dis_array = DistributedArray(
                device_mesh=self,
                aval=avals[i],
                sharding_spec=sharding_specs[i],
                remote_buffers=bufs[i],
                indices=indices[i]
            )
            ret.append(dis_array)

        return ret

    def _execute_with_handler(self, remote_executable, args_handler, outs_handler, num_outs, *args):
        num_args = len(args)

        # Shape: (num_args, total_devices)
        input_bufs = args_handler(args)

        # Shape: (num_hosts, num_args, num_devices_per_host)
        input_bufs = np.array(input_bufs)\
            .reshape(num_args, self.num_hosts, self.num_devices_per_host)\
            .transpose([1, 0, 2])

        # Allocate output buffer references
        # Shape: (num_hosts, num_outs, num_devices_per_host)
        output_bufs = np.empty(
            (self.num_hosts, num_outs, self.num_devices_per_host), dtype=object)
        for i in range(self.num_hosts):
            for j in range(num_outs):
                for k in range(self.num_devices_per_host):
                    output_bufs[i][j][k] = RemoteBufferRef(self, i, k)

        # Execute SPMD binary
        for i in range(self.num_hosts):
            host_inputs = get_uuid_np_array(input_bufs[i])
            host_outputs = get_uuid_np_array(output_bufs[i])
            self.workers[i].execute.remote(remote_executable.uuid, host_inputs, host_outputs)

        # Gather outputs
        # Shape: (num_outs, total_devices)
        output_bufs = output_bufs.transpose([1, 0, 2]).reshape((num_outs, self.total_devices))
        return outs_handler(output_bufs)

    def shutdown(self):
        self.shutdown_workers()
        for worker in self.workers:
            ray.kill(worker)
        self.workers = None


class MeshHostWorker:
    """A ray actor to manage the xla computation on a single host."""

    def __init__(self, server_address, node_id):
        self.node_id = node_id
        self.distributed_client = \
                xla_client._xla.get_distributed_runtime_client(server_address, node_id)
        self.distributed_client.connect()
        self.backend = xla_client._gpu_backend_factory(
                self.distributed_client, node_id=node_id)

        self.local_devices = self.backend.local_devices()
        self.local_buffers = {}    # Dict[uuid -> DeviceArray]
        self.executable = {}       # Dict[uuid -> Executable]

    def put_buffer(self, uuid, device_id, data):
        self.local_buffers[uuid] = \
            self.backend.buffer_from_pyval(data, self.local_devices[device_id])

    def get_buffers(self, uuids):
        if isinstance(uuids, Iterable):
            return [self.local_buffers[uuid] for uuid in uuids]
        return self.local_buffers[uuids]

    def delete_buffers(self, uuids):
        if isinstance(uuids, Iterable):
            for uuid in uuids:
                del self.local_buffers[uuid]
        else:
            del self.local_buffers[uuids]

    def delete_executable(self, uuid):
        self.executable[uuid].delete()
        del self.executable[uuid]

    def compile_executable(self,
                           uuid,
                           hlo_proto,
                           logical_mesh_shape,
                           auto_sharding_strategy_vector,
                           is_tuple_args):
        backend = self.backend
        num_devices = np.prod(logical_mesh_shape)

        assert num_devices == len(backend.devices())

        compile_options = xla_bridge.get_compile_options(
            num_replicas=1,
            num_partitions=num_devices,
            device_assignment=np.arange(num_devices).reshape((1, -1)),
            use_spmd_partitioning=True,
        )
        compile_options.parameter_is_tupled_arguments = is_tuple_args

        computation = xla_client.XlaComputation(hlo_proto)
        with XlaPassContext({
            "auto_sharding::enable": True,
            "auto_sharding::load_strategy": True,
            "auto_sharding::strategy_vector": to_int_tuple(auto_sharding_strategy_vector),

            # Device mesh
            "auto_sharding::device_mesh_ids": tuple(range(num_devices)),
            "auto_sharding::device_mesh_shape": tuple(logical_mesh_shape),

            # Other useless but required arguments
            "auto_sharding::device_mesh_alpha": (1.0,) * len(logical_mesh_shape),
            "auto_sharding::device_mesh_beta": (1.0,) * len(logical_mesh_shape),
        }):
            compiled_computation = backend.compile(computation, compile_options)

        self.executable[uuid] = compiled_computation

    def execute(self, executable_uuid, input_uuids, output_uuids):
        # Map uuids to input buffers
        device_inputs = [[None for _ in range(input_uuids.shape[1])]
            for _ in range(input_uuids.shape[0])]
        for i in range(input_uuids.shape[0]):
            for j in range(input_uuids.shape[1]):
                device_inputs[i][j] = self.local_buffers[input_uuids[i][j]]

        # Execute the executable
        device_outs = self.executable[executable_uuid].\
            execute_sharded_on_local_devices(device_inputs)

        # Store output buffers
        for i in range(output_uuids.shape[0]):
            for j in range(output_uuids.shape[1]):
                self.local_buffers[output_uuids[i][j]] = device_outs[i][j]

    def sync(self):
        return

    def shutdown(self):
        self.distributed_client.shutdown()


class DeviceCluster:
    """A ray cluster with gpu devices."""

    def __init__(self):
        from ray.worker import _global_node as ray_global_node
        self.head_info = ray_global_node.address_info
        self.head_ip = self.head_info['node_ip_address']

        # Gather host ids
        self.host_info = []
        for node in ray.nodes():
            for key in node["Resources"]:
                if key.startswith("node:"):
                    self.host_info.append(node)

        # Gather device info
        self.num_devices = []
        for host_info in self.host_info:
            number = host_info["Resources"]["GPU"]
            assert number.is_integer()
            self.num_devices.append(int(number))

    def get_physical_mesh(self, host_ids=None, num_devices_per_host=None):
        """
        Slice a subset of hosts and devices to form a physical device mesh.

        Args:
            host_ids: List[int]. The index of host nodes.
                'None' means using all hosts
            num_devices_per_host: int. The number of devices per host.
                'None' means using all devices

        Return:
            A physical multi-host device mesh
        """
        host_ids = host_ids or np.arange(len(self.host_info))
        host_info = [self.host_info[x] for x in host_ids]

        num_devices_per_host = num_devices_per_host or self.num_devices[host_ids[0]]
        for host_id in host_ids:
            assert self.num_devices[host_id] >= num_devices_per_host

        return MultiHostDeviceMesh(host_ids, host_info, num_devices_per_host, self.head_ip)


########################################
# Register ShardArg Handler
########################################
def _device_mesh_put(device_mesh, shards):
    # Put shards to the distributed device
    buf_refs = []
    pt = 0
    for host_id in range(device_mesh.num_hosts):
        for device_id in range(device_mesh.num_devices_per_host):
            buf_ref = RemoteBufferRef(device_mesh, host_id, device_id)
            device_mesh.workers[host_id].put_buffer.remote(
                buf_ref.uuid, device_id, shards[pt])
            buf_refs.append(buf_ref)
            pt += 1
    return buf_refs

def _shard_array(x, device_mesh, indices):
    return _device_mesh_put(device_mesh, [x[i] for i in indices])

def _shard_device_array(array, device_mesh, indices):
    # Create shards according to indices
    start_indices, limit_indices, removed_dims = map(tuple, unzip3(
        _as_slice_indices(array, idx) for idx in indices))
    shards = array._multi_slice(start_indices, limit_indices, removed_dims)

    return _device_mesh_put(device_mesh, shards)

def _shard_distributed_array(array, device_mesh, indices):
    return shard_arg_handlers[type(array._value)](array._value, device_mesh, indices)

shard_arg_handlers = {}  # Shard an argument to a distributed device mesh
for t in array_types:
    shard_arg_handlers[t] = _shard_array
shard_arg_handlers[xla._DeviceArray] = _shard_device_array
shard_arg_handlers[xla._CppDeviceArray] = _shard_device_array
shard_arg_handlers[DistributedArray] = _shard_distributed_array
shard_arg_handlers[ShardedDeviceArray] = _shard_distributed_array

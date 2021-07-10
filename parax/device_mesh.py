"""Cluster related configurations (e.g., topology)."""
import time
import logging
from collections.abc import Iterable
from collections import defaultdict
import pickle
from typing import Union, List

from operator import attrgetter
import numpy as np
import ray
from jax import core, xla
from jax._src.util import (partial, unzip3)
from jax.abstract_arrays import array_types
from jax.interpreters import pxla
from jax.interpreters.pxla import (ShardingSpec, Chunked, NoSharding, Replicated,
                                   ShardedAxis, _as_slice_indices, _hashable_index,
                                   ShardedDeviceArray)
from jax.lib import xla_client, xla_bridge

from parax.measure_record import StrategyConfig
from parax.global_env import global_config
from parax.profile_communication import profile_collective_one_config, ProfilingResult
from parax.util import (get_dim_last_value, list_gpu_info, profile_xla_executable, GB)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def device_id_to_str(host_ip, device_id, device_type="gpu"):
    """Convert device id (int) to a canonical device string."""
    return "{}:{}:{}".format(host_ip, device_type, str(device_id))


def device_str_to_id(device_str):
    """Parse device string to get its device id."""
    return int(device_str.split(":")[-1])


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
        sharding = [NoSharding(), ] * len(shape)
        mesh_mapping = [None, ] * len(self.id_mesh.shape)

        for i, (tensor_dim, mesh_dim) in enumerate(zip(tensor_dims, mesh_dims)):
            sharding[tensor_dim] = Chunked([self.id_mesh.shape[mesh_dim]], )
            mesh_mapping[mesh_dim] = ShardedAxis(i)

        for i, _ in enumerate(mesh_mapping):
            if mesh_mapping[i] is None:
                mesh_mapping[i] = Replicated(self.id_mesh.shape[i])

        return ShardingSpec(sharding, mesh_mapping)

    def __hash__(self):
        return hash((self.flatten_ids, self.id_mesh.shape,
                     self.mesh_alpha, self.mesh_beta))

    def __eq__(self, other):
        return (self.flatten_ids, self.id_mesh.shape,
                self.mesh_alpha, self.mesh_beta) == \
               (other.flatten_ids, other.id_mesh.shape,
                other.mesh_alpha, other.mesh_beta)


class RemoteExecutableRef:
    """A reference to a remote compiled XLA executable binary."""

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
    """A reference to a remote device buffer."""

    ct = 0

    def __init__(self, device_mesh, host_id, device_id):
        self.device_mesh = device_mesh
        self.host_id = host_id
        self.device_id = device_id
        self.uuid = RemoteBufferRef.ct
        self.is_donated = False
        RemoteBufferRef.ct = (RemoteBufferRef.ct + 1) % (1 << 60)
        logger.debug("RemoteBufferRef uuid: {} created on mesh with devices {}."
                     .format(self.uuid, self.device_mesh.devices_str))

    def donate(self):
        """
        Set the buffer as donated.

        If the buffer is donated, we do not need to explicitly call the actor to
        delete it. Its memory will be deleted by xla runtime.
        """
        self.is_donated = True

    def __repr__(self):
        return f"RemoteBufferRef(uuid = {self.uuid}, loc = ({self.host_id}, {self.device_id}))"

    def __del__(self):
        if not self.is_donated:
            self.device_mesh.delete_remote_buffers((self,))


class DistributedArray:
    """A distributed array on a PhysicalDeviceMesh."""

    def __init__(self, device_mesh, aval, sharding_spec, remote_buffers, indices):
        self.device_mesh = device_mesh
        self.aval = aval
        self.sharding_spec = sharding_spec
        self.remote_buffers = remote_buffers
        self.indices = indices

        self._npy_value = None
        self._one_replica_buffer_indices = None

    def block_until_ready(self):
        self.device_mesh.block_until_ready_remote_buffers(self.remote_buffers)

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
            for ct, i in enumerate(self.one_replica_buffer_indices):
                npy_value[self.indices[i]] = fetched_np_buffers[ct]
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


class MeshHostWorker:
    """A ray actor to manage the xla computation on a single host."""

    def __init__(self, server_address, num_hosts, host_id):
        self.num_hosts = num_hosts
        self.host_id = host_id
        self.distributed_client = \
            xla_client._xla.get_distributed_runtime_client(server_address, host_id)
        self.distributed_client.connect()
        self.backend = xla_client.make_gpu_client(
            self.distributed_client, node_id=host_id)

        self.local_devices = self.backend.local_devices()
        self.local_buffers = {}  # Dict[uuid -> DeviceArray]
        self.executables = {}  # Dict[uuid -> Executable]

    ##### Buffer Related Functions #####
    def put_buffer(self, uuid: int, device_id: int, data: np.ndarray):
        self.local_buffers[uuid] = \
            self.backend.buffer_from_pyval(data, self.local_devices[device_id])

    def put_dummy_buffer(self, uuid: int, device_id: int, shape, dtype):
        self.local_buffers[uuid] = \
            self.backend.buffer_from_pyval(np.empty(shape, dtype),
                                           self.local_devices[device_id])

    def get_buffers(self, uuids: Union[List[int], int]):
        if isinstance(uuids, Iterable):
            return [self.local_buffers[uuid] for uuid in uuids]
        return self.local_buffers[uuids]

    def delete_buffers(self, uuids: Union[List[int], int]):
        if isinstance(uuids, Iterable):
            for uuid in uuids:
                del self.local_buffers[uuid]
        else:
            del self.local_buffers[uuids]

    def block_until_ready_buffers(self, uuids: Union[List[int], int]):
        if isinstance(uuids, Iterable):
            for uuid in uuids:
                self.local_buffers[uuid].block_until_ready()
        else:
            self.local_buffers[uuids].block_until_ready()

    ##### Executable Related Functions #####
    def delete_executable(self, uuid: int):
        self.executables[uuid].delete()
        del self.executables[uuid]

    def compile_executable(self,
                           uuid: int,
                           hlo_proto: bytes,
                           strategy_config: StrategyConfig,
                           hlo_proto_is_sharded: bool):
        # pylint: disable=import-outside-toplevel
        from parax.auto_sharding import compile_with_given_strategy

        xla_computation = xla_client.XlaComputation(hlo_proto)
        num_devices = np.prod(strategy_config.logical_mesh_shape)
        assert num_devices == len(self.backend.devices())

        compiled = compile_with_given_strategy(
            self.backend, xla_computation, strategy_config, num_devices,
            False, xla_computation_is_sharded=hlo_proto_is_sharded)
        self.executables[uuid] = compiled

    def execute(self,
                executable_uuid: int,
                input_uuids: List[List[int]],
                output_uuids: List[List[int]]):
        # Map uuids to input buffers
        device_inputs = [[None for _ in range(input_uuids.shape[1])]
                         for _ in range(input_uuids.shape[0])]
        for i in range(input_uuids.shape[0]):
            for j in range(input_uuids.shape[1]):
                device_inputs[i][j] = self.local_buffers[input_uuids[i][j]]

        # Execute the executable
        device_outs = self.executables[executable_uuid]. \
            execute_sharded_on_local_devices(device_inputs)

        # Store output buffers
        for i in range(output_uuids.shape[0]):
            for j in range(output_uuids.shape[1]):
                self.local_buffers[output_uuids[i][j]] = device_outs[i][j]

        # Delete donated input buffers
        for i in range(input_uuids.shape[0]):
            for j in range(input_uuids.shape[1]):
                if device_inputs[i][j].is_deleted():
                    del self.local_buffers[input_uuids[i][j]]

    ##### Profiling Related Functions #####
    def profile_collective(self, primitive_name, size_range, number, verbose):
        """Profile the time cost of collective communication primitive (all-reduce, all-gather)."""
        # Generate all possible communication groups
        prof_result = ProfilingResult()
        size_configs = []
        size_configs.append((0, "float32"))
        for i in size_range or range(30):
            size_configs.append((1 << i, "float32"))

        logical_mesh_shapes = []
        total_devices = self.num_hosts * len(self.local_devices)
        for i in range(1, total_devices + 1):
            if total_devices % i == 0:
                logical_mesh_shapes.append((total_devices // i, i))

        all_keys = set()
        for logical_mesh_shape in logical_mesh_shapes:
            # dim 0
            replica_groups = []
            tmp_group = []
            for i in range(logical_mesh_shape[0]):
                tmp_group.append(
                    tuple(i * logical_mesh_shape[1] + j for j in range(logical_mesh_shape[1])))
            replica_groups.append(tuple(tmp_group))

            # dim 1
            tmp_group = []
            for j in range(logical_mesh_shape[1]):
                tmp_group.append(
                    tuple(i * logical_mesh_shape[1] + j for i in range(logical_mesh_shape[0])))
            replica_groups.append(tuple(tmp_group))

            for replica_group in replica_groups:
                for size, dtype in size_configs:
                    all_keys.add((replica_group, size, dtype))
        all_keys = list(all_keys)
        all_keys.sort()

        for replica_group, size, dtype in all_keys:
            if number == "auto":
                number_ = min(max(15, int((1 << 31) / (max(size, 1) * np.dtype(dtype).itemsize))),
                              1 << 13)
            else:
                number_ = number

            time_cost = profile_collective_one_config(
                (size,), dtype, replica_group, primitive_name,
                self.backend, self.num_hosts * len(self.local_devices),
                self.local_devices, self.distributed_client, self.host_id,
                self.sync, number_)

            num_devices = len(replica_group[0])
            array_size = size * np.dtype(dtype).itemsize

            if primitive_name == "all-reduce":
                prof_result.record_all_reduce(replica_group, size, dtype, time_cost)
                communication_size = 2 * array_size * (num_devices - 1) / num_devices
            elif primitive_name == "all-gather":
                prof_result.record_all_gather(replica_group, size, dtype, time_cost)
                communication_size = array_size * (num_devices - 1) / num_devices
            else:
                raise ValueError("Invalid primitive: " + primitive_name)

            bandwidth = communication_size / time_cost

            if self.host_id == 0 and verbose >= 1:
                heads = [primitive_name, "Size (GB)", "Time", "Bandwidth (GB/s)"]
                values = [str(replica_group), f"{array_size / GB:.5f}",
                          f"{time_cost:.5f}", f"{bandwidth / GB:.2f}"]

                line = ""
                for head, value in zip(heads, values):
                    line += head + ": " + value + "  "
                print(line)

        if self.host_id == 0:
            return prof_result
        return None

    def profile_executable(self, executable_uuid: int):
        return profile_xla_executable(self.executables[executable_uuid], self.backend,
                                      self.local_devices, self.sync)

    ##### Other Functions #####
    def sync(self):
        for device in self.local_devices:
            device.synchronize_all_activity()

    def shutdown(self):
        self.sync()
        del self.local_buffers
        del self.executables
        self.distributed_client.shutdown()


class PhysicalDeviceMesh:
    """Class for either single-host or multi-host physical device mesh."""

    def __init__(self,
                 devices=None,
                 host_ids=None,
                 host_info=None,
                 head_ip=None,
                 num_devices_per_host=1,
                 use_ray=False):
        # actually we can infer use_ray by checking ip addresses.
        self.use_ray = use_ray
        self.host_ids = host_ids
        self.host_info = host_info
        self.head_ip = head_ip
        self.num_devices_per_host = num_devices_per_host
        self.workers = None
        self.prof_result = ProfilingResult()

        # Do some argument check
        if not use_ray and not devices:
            raise RuntimeError("`devices` are required for single-host device mesh.")
        if devices and use_ray:
            raise RuntimeError("`devices` should not be passed in when using a Ray cluster.")
        if not use_ray:
            self.devices = devices
            self.host_ids = [0]
            self.host_info = None
            self.head_ip = "127.0.0.1"
            self.devices_str = [device_id_to_str(self.head_ip, d.id) for d in devices]
            self.num_devices_per_host = len(self.devices)

        if use_ray:
            self.devices_str = []
            for i, _ in enumerate(self.host_ids):
                ip = self.host_info[i]["NodeManagerAddress"]
                self.devices_str.extend([device_id_to_str(ip, i)
                                         for i in range(self.num_devices_per_host)])

            self._launch_xla_servers()

    def _launch_xla_servers(self):
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
            # Set XLA environment variables
            env_vars = {
                #"XLA_FLAGS": "--xla_gpu_autotune_level=0",
                #"XLA_FLAGS": "--xla_dump_to=hlo --xla_dump_hlo_pass_re=.*"
            }

            # Launch a ray actor
            node_resource = "node:" + self.host_info[i]["NodeManagerAddress"]
            cls = ray.remote(num_gpus=self.num_devices_per_host,
                             resources={node_resource: 1e-3})(MeshHostWorker)
            worker = cls.options(override_environment_variables=env_vars).remote(
                self.server_address, self.num_hosts, i)
            self.workers.append(worker)
        self.sync_workers()

    def get_signature(self) -> str:
        """Return a signature string that contains the mesh shape and GPU model."""
        gpu_type = list_gpu_info()
        gpu_name = gpu_type.split("\n")[0].split(" (UUID:")[0][7:]
        ret = f"{len(self.host_ids)},{self.num_devices_per_host},{gpu_name}"
        ret = ret.replace(" ", "-")
        return ret

    @property
    def total_devices(self):
        """Return the total number of GPUs on this mesh."""
        return len(self.device_ids)

    @property
    def num_hosts(self):
        """Return the number of hosts in the mesh."""
        return len(self.host_ids)

    @property
    def device_ids(self):
        """Return the device ids (does not distinguish host IPs)."""
        return [device_str_to_id(device_str) for device_str in self.devices_str]

    @property
    def is_distributed(self):
        """Whether this mesh should be considered as a distributed mesh."""
        return not (self.num_hosts == 1 and not self.use_ray)

    ##### Mesh Related Functions #####
    def get_logical_mesh(self,
                         mesh_shape,
                         mesh_alpha=None,
                         mesh_beta=None,
                         mesh_topology=None,
                         intra_host_bandwidth=None,
                         inter_host_bandwidth=None):
        """Return a logical mesh and parameters of the alpha-beta communication cost model."""
        id_mesh = np.arange(self.total_devices).reshape(mesh_shape)

        if mesh_topology is None:
            mesh_alpha = mesh_alpha or (1,) * len(mesh_shape)
            mesh_beta = mesh_beta or (1,) * len(mesh_shape)
        elif mesh_topology == "tree":
            assert mesh_alpha is None
            assert mesh_beta is None
            mesh_alpha = [1] * 2
            mesh_beta = [None] * 2
            host_ids = np.tile(np.arange(self.num_hosts).reshape(-1, 1),
                               self.num_devices_per_host)
            host_ids = host_ids.reshape(mesh_shape)

            # Compute bandwidth of doing communication along dim 0.
            # 1. Compute the number of links between each host pairs.
            #    Assume using ring-based algorithms.
            host_link_ct = defaultdict(int)
            for j in range(mesh_shape[1]):
                for i in range(mesh_shape[0]):
                    left = host_ids[i][j]
                    right = host_ids[(i + 1) % mesh_shape[0]][j]
                    if left != right:
                        if left > right:
                            left, right = right, left
                        host_link_ct[(left, right)] += 1

            j = 0
            # 2. Bandwidth between two hosts = total_bandwidth / number_of_links.
            #    Bandwdith along a communication dimension = min bandwidth of all links.
            bandwidth = intra_host_bandwidth
            for i in range(mesh_shape[0]):
                left = host_ids[i][j]
                right = host_ids[(i + 1) % mesh_shape[0]][j]
                if left != right:
                    if left > right:
                        left, right = right, left
                    bandwidth = min(bandwidth,
                                    inter_host_bandwidth / host_link_ct[(left, right)])
            mesh_beta[0] = 1 / bandwidth

            # Compute bandwidth of doing communication along dim 1.
            host_link_ct = defaultdict(int)
            for i in range(mesh_shape[0]):
                for j in range(mesh_shape[1]):
                    left = host_ids[i][j]
                    right = host_ids[i][(j + 1) % mesh_shape[1]]
                    if left != right:
                        if left > right:
                            left, right = right, left
                        host_link_ct[(left, right)] += 1

            i = 0
            bandwidth = intra_host_bandwidth
            for j in range(mesh_shape[1]):
                left = host_ids[i][j]
                right = host_ids[i][(j + 1) % mesh_shape[1]]
                if left != right:
                    if left > right:
                        left, right = right, left
                    bandwidth = min(bandwidth,
                                    inter_host_bandwidth / host_link_ct[(left, right)])
            mesh_beta[1] = 1 / bandwidth

        return LogicalDeviceMesh(self, id_mesh, mesh_alpha, mesh_beta)

    def get_default_logical_mesh(self):
        """Return the default logical mesh."""
        if self.num_hosts == 1:
            return self.get_logical_mesh((self.num_hosts, self.num_devices_per_host),
                                         [1, 1], [1, 1])
        else:
            return self.get_logical_mesh((self.num_hosts, self.num_devices_per_host),
                                         [1, 1], [1, 0.01])

    ##### Buffer Related Functions #####
    def get_remote_buffers(self, buf_refs: List[RemoteBufferRef]):
        obj_refs = []
        for buf_ref in buf_refs:
            obj_refs.append(self.workers[buf_ref.host_id].
                            get_buffers.remote(buf_ref.uuid))

        return ray.get(obj_refs)

    def delete_remote_buffers(self, buf_refs: List[RemoteBufferRef]):
        if self.workers is None or not ray.is_initialized():
            return

        for buf_ref in buf_refs:
            self.workers[buf_ref.host_id].delete_buffers.remote(buf_ref.uuid)

    def block_until_ready_remote_buffers(self, buf_refs: List[RemoteBufferRef]):
        tasks = []
        for buf_ref in buf_refs:
            tasks.append(self.workers[buf_ref.host_id].
                         block_until_ready_buffers.remote(buf_ref.uuid))
        ray.get(tasks)

    ##### Executable Related Functions #####
    def compile_remote_executable(self,
                                  hlo_proto: bytes,
                                  strategy_config: StrategyConfig,
                                  hlo_proto_is_sharded: bool):
        """Compile the remote executable."""
        executable = RemoteExecutableRef(self)
        for w in self.workers:
            w.compile_executable.remote(
                executable.uuid,
                hlo_proto,
                strategy_config,
                hlo_proto_is_sharded)
        return executable

    def delete_remote_executable(self, exe_ref: RemoteExecutableRef):
        if self.workers is None or not ray.is_initialized():
            return

        for i in range(self.num_hosts):
            self.workers[i].delete_executable.remote(exe_ref.uuid)

    def get_callable_with_arg_handler(self, compiled_executable, avals, out_avals,
                                      input_sharding_specs, output_sharding_specs,
                                      donated_invars):
        # pylint: disable=too-many-arguments
        """Get a callable after sharding optimization."""
        input_indices = [pxla.spec_to_indices(aval.shape, spec)
                         for aval, spec in zip(avals, input_sharding_specs)]

        args_handler = partial(self._shard_args, input_indices, donated_invars)
        if not self.is_distributed:
            outs_handler = pxla.avals_to_results_handler(1, len(self.devices),
                                                         output_sharding_specs, out_avals)
        else:
            output_indices = [pxla.spec_to_indices(aval.shape, spec) for
                              aval, spec in zip(out_avals, output_sharding_specs)]
            outs_handler = partial(self._gather_outs, out_avals, output_sharding_specs,
                                   output_indices)
        ret = partial(self._execute_with_handler, compiled_executable, args_handler,
                      outs_handler, len(out_avals), donated_invars)
        ret.shard_args_only = partial(self.preshard_args, args_handler, avals,
                                      input_sharding_specs, input_indices)
        return ret

    def _execute_with_handler(self, executable, args_handler, outs_handler,
                              num_outs, donated_invars, *args):
        # pylint: disable=too-many-arguments,too-many-locals
        input_bufs = args_handler(args)
        if not self.is_distributed:
            output_bufs = executable.execute_sharded_on_local_devices(input_bufs)
        else:
            # Donate input buffers
            for bufs, is_donated in zip(input_bufs, donated_invars):
                if is_donated:
                    for buf in bufs:
                        buf.donate()

            # Shape: (num_hosts, num_args, num_devices_per_host)
            input_bufs = np.array(input_bufs) \
                .reshape(len(args), self.num_hosts, self.num_devices_per_host) \
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
                self.workers[i].execute.remote(executable.uuid, host_inputs, host_outputs)
            # Gather outputs
            # Shape: (num_outs, total_devices)
            output_bufs = output_bufs.transpose([1, 0, 2]).reshape((num_outs, self.total_devices))
        return outs_handler(output_bufs)

    def preshard_args(self, handler, avals, sharding_specs, indices, *args):
        """Pre-shard the input arguments."""
        input_bufs = handler(args)
        sharded_args = []
        for i in range(len(args)):
            if self.is_distributed:
                array = DistributedArray(self, avals[i], sharding_specs[i], input_bufs[i], indices[i])
            else:
                array = ShardedDeviceArray(avals[i], sharding_specs[i], input_bufs[i], indices[i])
            sharded_args.append(array)
        return sharded_args

    def _shard_args(self, arg_indices, donated_invars, args):
        if not self.is_distributed:
            # single host w/o Ray
            return pxla.shard_args(self.devices, arg_indices, args)
        else:
            input_bufs = []
            for arg, indices, donated in zip(args, arg_indices, donated_invars):
                # Fast path for DistributedArray
                if isinstance(arg, DistributedArray) and arg.indices == indices:
                    input_bufs.append(arg.remote_buffers)
                else:  # Slow path
                    arg = xla.canonicalize_dtype(arg)
                    buf_refs = shard_arg_handlers[type(arg)](arg, self, indices)
                    input_bufs.append(buf_refs)
                    if donated and isinstance(arg, (xla._DeviceArray, xla._CppDeviceArray)):
                        arg.delete()

            return input_bufs

    def _gather_outs(self, avals, sharding_specs, indices, bufs):
        ret = []
        for i, _ in enumerate(avals):
            dis_array = DistributedArray(
                device_mesh=self,
                aval=avals[i],
                sharding_spec=sharding_specs[i],
                remote_buffers=bufs[i],
                indices=indices[i]
            )
            ret.append(dis_array)

        return ret

    ##### Profling related Functions #####
    def profile_collective(self, primitive_name, size_range=None, number="auto", verbose=1):
        """Profile the time cost of collective communication primitive (all-reduce, all-gather)."""
        tasks = []
        for worker in self.workers:
            tasks.append(worker.profile_collective.remote(
                primitive_name, size_range, number, verbose))
        prof_result = ray.get(tasks)[0]
        if primitive_name == "all-reduce":
            self.prof_result.all_reduce_cost_dict = prof_result.all_reduce_cost_dict
        elif primitive_name == "all-gather":
            self.prof_result.all_gather_cost_dict = prof_result.all_gather_cost_dict
        else:
            raise ValueError("Invalid primitive_name: " + primitive_name)

    def profile_executable(self, compiled, unoptimized_hlo_proto, strategy_config):
        """Profile the time cost of an xla executable."""
        if self.is_distributed:
            # Send the code and strategy to remote workers
            compiled = self.compile_remote_executable(
                unoptimized_hlo_proto, strategy_config, hlo_proto_is_sharded=False)

            # Run profiling
            tasks = []
            for worker in self.workers:
                tasks.append(worker.profile_executable.remote(compiled.uuid))
            costs = ray.get(tasks)[0]
        else:
            costs = profile_xla_executable(compiled, xla_bridge.get_backend("gpu"),
                                           self.devices, self.sync_workers)

        return costs

    def load_profiling_result(self, filename):
        """Load profiling results from a file."""
        self.prof_result = pickle.load(open(filename, "rb"))

    def save_profiling_result(self, filename):
        """Save profiling results to a file."""
        pickle.dump(self.prof_result, open(filename, "wb"))

    ##### Other Functions #####
    def sync_workers(self):
        """Sync all device activities on workers."""
        if self.is_distributed:
            ray.get([w.sync.remote() for w in self.workers])
        else:
            for device in self.devices:
                device.synchronize_all_activity()

    def shutdown(self):
        """Shut down the mesh."""
        if self.is_distributed:
            ray.get([w.shutdown.remote() for w in self.workers])
            for worker in self.workers:
                ray.kill(worker)
            self.workers = None
        else:
            self.sync_workers()


# TODO (Hao): merge VirtualMesh into PhysicalMesh by adding a start_cluster attribute.
class VirtualMesh:
    """A virtual mesh used to instantiate a Physical Mesh in the future."""

    def __init__(self,
                 *,
                 host_ids=None,
                 host_info=None,
                 head_ip=None,
                 num_devices_per_host=1):
        self.host_ids = host_ids
        self.host_info = host_info
        self.head_ip = head_ip
        self.num_devices_per_host = num_devices_per_host

        self.devices_str = []
        for i, _ in enumerate(self.host_ids):
            ip = self.host_info[i]["NodeManagerAddress"]
            self.devices_str.extend([device_id_to_str(ip, i)
                                     for i in range(self.num_devices_per_host)])

    def slice(self, dim, indices):
        """
        Slice a mesh given the slicing config.

        Args:
            dim (int): which dimension to slice from, num_host or num_gpu
            indices (List[int]):

        Returns:
            mesh (PhysicalDeviceMesh)
        """
        if dim == 0:
            # slicing along the host dimension
            host_ids = [self.host_ids[x] for x in indices]
            host_info = [self.host_info[x] for x in host_ids]
            return VirtualMesh(host_ids=host_ids, host_info=host_info,
                               head_ip=self.head_ip, num_devices_per_host=self.num_devices_per_host)
        else:
            # slicing along the device dimension
            return VirtualMesh(host_ids=self.host_ids, host_info=self.host_info,
                               head_ip=self.head_ip, num_devices_per_host=len(indices))

    @property
    def total_devices(self):
        """Return the total number of GPUs on this mesh."""
        return len(self.device_ids)

    @property
    def num_hosts(self):
        """Return the number of hosts in the mesh."""
        return len(self.host_ids)

    @property
    def device_ids(self):
        """Return the device ids (does not distinguish host IPs)."""
        return [device_str_to_id(device_str) for device_str in self.devices_str]

    @property
    def is_distributed(self):
        """Whether this mesh should be considered as a distributed mesh."""
        return True

    def get_physical_mesh(self):
        """Convert to a physical mesh (which will request resources from Ray)."""
        return PhysicalDeviceMesh(host_ids=self.host_ids,
                                  host_info=self.host_info,
                                  head_ip=self.head_ip,
                                  num_devices_per_host=self.num_devices_per_host,
                                  use_ray=True)

    def get_logical_mesh(self, mesh_shape, mesh_alpha=None, mesh_beta=None):
        """Generate a logical mesh."""
        id_mesh = np.arange(self.total_devices).reshape(mesh_shape)
        mesh_alpha = mesh_alpha or (1.0,) * len(mesh_shape)
        mesh_beta = mesh_beta or (1.0,) * len(mesh_shape)
        return LogicalDeviceMesh(self, id_mesh, mesh_alpha, mesh_beta)

    def get_default_logical_mesh(self):
        """Return the default logical mesh."""
        if self.num_hosts == 1:
            return self.get_logical_mesh((self.num_hosts, self.num_devices_per_host),
                                         [1, 1], [1, 1])
        else:
            return self.get_logical_mesh((self.num_hosts, self.num_devices_per_host),
                                         [1, 1], [1, 0.1])


class DeviceCluster:
    """A ray cluster with GPU devices."""

    def __init__(self):
        # pylint: disable=import-outside-toplevel
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

        return PhysicalDeviceMesh(host_ids=host_ids,
                                  host_info=host_info,
                                  num_devices_per_host=num_devices_per_host,
                                  head_ip=self.head_ip,
                                  use_ray=True)

    def get_virtual_mesh(self,
                         host_ids=None,
                         num_devices_per_host=None):
        """
        Slice a subset of hosts and devices to form a virtual device mesh.

        The only difference between a virtual and a physical mesh is that a virtual
        mesh does not request cluster resources.
        """
        host_ids = host_ids or np.arange(len(self.host_info))
        host_info = [self.host_info[x] for x in host_ids]

        num_devices_per_host = num_devices_per_host or self.num_devices[host_ids[0]]
        for host_id in host_ids:
            assert self.num_devices[host_id] >= num_devices_per_host

        return VirtualMesh(host_ids=host_ids,
                           host_info=host_info,
                           num_devices_per_host=num_devices_per_host,
                           head_ip=self.head_ip)


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
            if global_config.use_dummy_value_for_benchmarking:
                device_mesh.workers[host_id].put_dummy_buffer.remote(
                    buf_ref.uuid, device_id, shards[pt].shape, shards[pt].dtype)
            else:
                device_mesh.workers[host_id].put_buffer.remote(
                    buf_ref.uuid, device_id, shards[pt])
            buf_refs.append(buf_ref)
            pt += 1
    return buf_refs


def _shard_array(x, device_mesh, indices):
    # Create shards according to indices for a numpy array
    return _device_mesh_put(device_mesh, [x[i] for i in indices])


def _shard_device_array(array, device_mesh, indices):
    # Create shards according to indices for a DeviceArray
    start_indices, limit_indices, removed_dims = map(tuple, unzip3(
        _as_slice_indices(array, idx) for idx in indices))
    shards = array._multi_slice(start_indices, limit_indices, removed_dims)

    return _device_mesh_put(device_mesh, shards)


def _shard_distributed_array(array, device_mesh, indices):
    # Create shards according to indices for a DistributedArray
    return shard_arg_handlers[type(array._value)](array._value, device_mesh, indices)


shard_arg_handlers = {}  # Shard an argument to a distributed device mesh
for t in array_types:
    shard_arg_handlers[t] = _shard_array
shard_arg_handlers[xla._DeviceArray] = _shard_device_array
shard_arg_handlers[xla._CppDeviceArray] = _shard_device_array
shard_arg_handlers[DistributedArray] = _shard_distributed_array
shard_arg_handlers[ShardedDeviceArray] = _shard_distributed_array


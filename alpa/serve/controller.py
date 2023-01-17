#pylint: disable=missing-class-docstring, raise-missing-from
"""Central controller"""
import asyncio
from collections import defaultdict
import dataclasses
import logging
import os
import pickle
import socket
import time
from typing import Callable, List, Dict, Optional, Tuple, Any, Union

import ray
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from starlette.middleware.cors import CORSMiddleware
import uvicorn

from alpa.api import init
from alpa.serve.http_util import (HTTPRequestWrapper, receive_http_body,
                                  Response, set_socket_reuse_port, ASGIHandler,
                                  build_starlette_request, new_port,
                                  RelayException, make_error_response)

logger = logging.getLogger(__file__)

CONTROLLER_NAME = "controller"
MAX_REPLICA_FAILURE_RETRIES = 10
DISCONNECT_ERROR_CODE = "disconnection"
SOCKET_REUSE_PORT_ENABLED = (os.environ.get("SERVE_SOCKET_REUSE_PORT_ENABLED",
                                            "1") == "1")


@dataclasses.dataclass
class CreateInfo:
    model_def: Any
    init_args: Optional[List]
    init_kwargs: Optional[Dict]

    def append_init_args(self,
                         init_args: Optional[List] = None,
                         init_kwargs: Optional[Dict] = None):
        return CreateInfo(
            self.model_def,
            self.init_args + init_args if init_args else self.init_args,
            dict(self.init_kwargs).update(init_kwargs)
            if init_kwargs else self.init_kwargs,
        )


@dataclasses.dataclass
class ModelInfo:
    create_info: CreateInfo
    managers: List[ActorHandle]
    next_pt: int


@ray.remote(num_cpus=1)
class DeviceMeshGroupManager:

    def __init__(self, virtual_mesh_shape: Optional[Tuple[int]] = None):
        if virtual_mesh_shape:
            init(cluster="ray",
                 num_nodes=virtual_mesh_shape[0],
                 num_devices_per_node=virtual_mesh_shape[1])
        else:
            init(cluster="ray")

        # Dict[str, object]
        self.replicas = {}

    def create_replica(self, name: str, create_info: CreateInfo):
        assert name not in self.replicas

        model_def, args, kwargs = (create_info.model_def, create_info.init_args,
                                   create_info.init_kwargs)
        args = args or []
        kwargs = kwargs or {}
        self.replicas[name] = model_def(*args, **kwargs)

    def delete_replica(self, name: str):
        assert name in self.replicas
        del self.replicas[name]

    async def handle_request(self, name: str, request_wrapper: bytes):
        request_wrapper = pickle.loads(request_wrapper)
        request = build_starlette_request(request_wrapper)
        try:
            response = await self.replicas[name].handle_request(request)
            return response
        except Exception as e:  # pylint: disable=broad-except
            return RelayException(e)


@ray.remote(num_cpus=0)
class Controller:

    def __init__(self,
                 host: str,
                 port: int,
                 root_path: str,
                 ssl_keyfile: Optional[str] = None,
                 ssl_certfile: Optional[Union[str, os.PathLike]] = None):
        self.host = host
        self.port = port
        self.root_path = root_path
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile

        self.manager_lock = defaultdict(asyncio.Lock)

        # Dict[str -> ModelInfo]
        self.model_info = {}
        self.mesh_group_managers = {}

        # Launch http server
        self.setup_complete = asyncio.Event()
        self.http_server_task = asyncio.get_event_loop().create_task(
            self.run_http_server())

    async def launch_mesh_group_manager(
            self,
            group_id: int,
            virtual_mesh_shape: Optional[Tuple[int]] = None,
            num_gpus: int = 0):
        assert group_id not in self.mesh_group_managers, (
            f"Mesh group {group_id} is already launched")
        self.mesh_group_managers[group_id] = (DeviceMeshGroupManager.options(
            name=f"mesh_group_manager_{group_id}",
            num_gpus=num_gpus).remote(virtual_mesh_shape))

    async def register_model(self,
                             name: str,
                             model_def: Callable,
                             init_args: Optional[List] = None,
                             init_kwargs: Optional[Dict] = None,
                             override: bool = False):
        async with self.manager_lock[name]:
            if name in self.model_info:
                if override:
                    for manager in self.model_info[name].managers:
                        await manager.delete_replica.remote(name)
                else:
                    raise ValueError(f"Model {name} is already registered")

            self.model_info[name] = ModelInfo(
                CreateInfo(model_def, init_args, init_kwargs), [], 0)

    async def create_replica(self,
                             name: str,
                             mesh_group_id: int,
                             append_init_args: Optional[List] = None,
                             append_init_kwargs: Optional[Dict] = None):
        async with self.manager_lock[name]:
            assert mesh_group_id in self.mesh_group_managers, (
                f"Group {mesh_group_id} does not exist")
            model_info = self.model_info[name]
            manager = self.mesh_group_managers[mesh_group_id]
            assert manager not in model_info.managers
            create_info = model_info.create_info.append_init_args(
                append_init_args, append_init_kwargs)

            logger.info(
                f"Create replica of model={name} on mesh={mesh_group_id}")
            await manager.create_replica.remote(name, create_info)
            model_info.managers.append(manager)

    async def handle_asgi(self, scope, receive, send):
        assert scope["type"] == "http"
        scope["tstamp"] = time.time()

        # Receive request
        http_body_bytes = await receive_http_body(scope, receive, send)
        request_wrapper = HTTPRequestWrapper(scope, http_body_bytes)
        request = build_starlette_request(request_wrapper)
        request_wrapper = pickle.dumps(request_wrapper)

        # Route
        try:
            obj = await request.json()

            assert "model" in obj, "Model name is not specified in the request."
            name = obj["model"]

            assert name in self.model_info, (
                f"Model '{name}' is not registered.")
            model_info = self.model_info[name]
            assert model_info.managers, (
                f"No replica of model '{name}' is created.")
            manager = model_info.managers[model_info.next_pt]
            model_info.next_pt = (model_info.next_pt + 1) % len(
                model_info.managers)

            response = await manager.handle_request.remote(
                name, request_wrapper)
            if isinstance(response, RelayException):
                response = make_error_response(response)
                status_code = 400
            else:
                status_code = 200
        except Exception as e:  # pylint: disable=broad-except
            response = make_error_response(e)
            status_code = 400

        await Response(response,
                       status_code=status_code).send(scope, receive, send)

    def get_info(self):
        return {
            "host": self.host,
            "port": self.port,
            "root_path": self.root_path,
        }

    ##### HTTP related functions #####
    async def ready(self):
        """Returns when HTTP proxy is ready to serve traffic.
        Or throw exception when it is not able to serve traffic.
        """
        done_set, _ = await asyncio.wait(
            [
                # Either the HTTP setup has completed.
                # The event is set inside self.run.
                self.setup_complete.wait(),
                # Or self.run errored.
                self.http_server_task,
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Return None, or re-throw the exception from self.running_task.
        return await done_set.pop()

    async def run_http_server(self):
        sock = socket.socket()
        if SOCKET_REUSE_PORT_ENABLED:
            set_socket_reuse_port(sock)

        try:
            sock.bind((self.host, self.port))
        except OSError:
            # The OS failed to bind a socket to the given host and port.
            raise ValueError(
                f"Failed to bind HTTP proxy to '{self.host}:{self.port}'."
                f"Please make sure your http-host and http-port are "
                f"specified correctly.")

        # Note(simon): we have to use lower level uvicorn Config and Server
        # class because we want to run the server as a coroutine. The only
        # alternative is to call uvicorn.run which is blocking.
        app = ASGIHandler(self)
        app = CORSMiddleware(
            app,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            root_path=self.root_path,
            lifespan="off",
            access_log=False,
            ssl_keyfile=self.ssl_keyfile,
            ssl_certfile=self.ssl_certfile,
        )
        server = uvicorn.Server(config=config)

        # TODO(edoakes): we need to override install_signal_handlers here
        # because the existing implementation fails if it isn't running in
        # the main thread and uvicorn doesn't expose a way to configure it.
        server.install_signal_handlers = lambda: None

        self.setup_complete.set()
        await server.serve(sockets=[sock])


def run_controller(host,
                   port=None,
                   root_path="/",
                   name=CONTROLLER_NAME,
                   ssl_keyfile: Optional[str] = None,
                   ssl_certfile: Optional[Union[str, os.PathLike]] = None):
    controller = Controller.options(
        name=name,
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().node_id,
            soft=False,
        )).remote(
            host=host,
            port=port or new_port(),
            root_path=root_path,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
        )
    ray.get(controller.ready.remote())
    return controller

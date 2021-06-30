"""pipeline stage definitions."""
import itertools as it
import logging
from dataclasses import dataclass, field
from typing import Sequence, List, Set, Any, Dict

from abc import ABC, abstractmethod
from jax import jit
from jax._src.util import partial, safe_map, extend_name_stack, wrap_name
from jax.core import Atom, Var, JaxprEqn, Jaxpr, ClosedJaxpr, jaxpr_as_fun
from jax.interpreters import xla
from jax.lib import xla_bridge as xb, xla_client as xc

# pylint: disable=redefined-builtin
from parax import testing
from parax.auto_sharding import compile_with_search, hlo_sharding_to_sharding_spec
from parax.measure_record import StrategyConfig
from parax.device_mesh import PhysicalDeviceMesh, VirtualMesh
from parax.util import get_compile_options

unsafe_map, map = map, safe_map  # type: ignore


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class PipelineStage(ABC):
    """
    Base class of pipeline stages.

    Attributes:
        name (str): The name of the pipeline stage.
        invars (Sequence[Var]): The list of input variables, corresponding to
            the order of the runnable inputs.
        pipeline_invars (Set[Var]): The set of input variables receiving from
            the previous pipeline stage.
        global_invars (Set[Var]): The set of input variables from driver
            function inputs.
        local_invars (Set[Var]): The set of input variables from previous
            stages running on the same device.
        outvars (Sequence[Var]): The list of output variables, corresponding to
            the order of the runnable outputs.
        pipeline_outvars (Set[Var]): The set of output variables sending to
            the next pipeline stage.
        global_outvars (Set[Var]): The set of output variables that will be
            used as driver function outputs.
        local_outvars (Set[Var]): The set of output variables that will be used
            by future stages running on the same device.
    """

    name: str
    # invars
    invars: Sequence[Var] = field(default_factory=list)
    pipeline_invars: Set[Var] = field(default_factory=set)
    global_invars: Set[Var] = field(default_factory=set)
    local_invars: Set[Var] = field(default_factory=set)
    # outvars
    outvars: Sequence[Var] = field(default_factory=list)
    pipeline_outvars: Set[Var] = field(default_factory=set)
    global_outvars: Set[Var] = field(default_factory=set)
    local_outvars: Set[Var] = field(default_factory=set)
    # intermediate vars
    intermediate_vars: Set[Var] = field(default_factory=set)

    @abstractmethod
    def get_runnable(self, mesh=None):
        """Compile the stage and get the runnable."""
        raise NotImplementedError()


@dataclass
class JaxPipelineStage(PipelineStage):
    """
    Pipeline stage with JaxPr.

    Attributes:
        eqns (List[JaxprEqn]): Jaxpr equations of the pipeline stage.
        consts_dir: Dict[Atom, Any]: All the constants used in the pipeline
            stage.
    """

    eqns: List[JaxprEqn] = field(default_factory=list)
    consts_dir: Dict[Atom, Any] = field(default_factory=dict)

    def closed_jaxpr(self) -> ClosedJaxpr:
        """
        Get the closed Jaxpr of the pipeline stage.

        Returns:
            ClosedJaxpr: The result ClosedJaxpr.
        """
        jaxpr = Jaxpr(
            constvars=self.consts_dir.keys(),
            invars=self.invars,
            outvars=self.outvars,
            eqns=self.eqns,
        )
        closed_jaxpr = ClosedJaxpr(jaxpr, self.consts_dir.values())
        return closed_jaxpr

    def get_runnable(self, mesh=None):
        """Return a JIT callable of the pipeline stage."""
        closed_jaxpr = self.closed_jaxpr()
        return jit(jaxpr_as_fun(closed_jaxpr))


@dataclass
class XlaPipelineStage(PipelineStage):
    """Pipeline stage with XLA HLO protos."""

    hlo_proto: bytes = field(default_factory=b"")

    @classmethod
    def from_jax_pipeline_stage(cls, jax_pipeline_stage: JaxPipelineStage):
        """
        Construct a XlaPipelineStage from a JaxPipelineStage.

        Args:
            jax_pipeline_stage (JaxPipelineStage): the source JaxPipelineStage.
        """
        closed_jaxpr = jax_pipeline_stage.closed_jaxpr()
        in_avals = [var.aval for var in jax_pipeline_stage.invars]
        consts = closed_jaxpr.consts
        map(xla.prefetch, it.chain(consts, xla.jaxpr_literals(closed_jaxpr.jaxpr)))

        backend = 'gpu'
        tuple_args = len(in_avals) > 100  # pass long arg lists as tuple for TPU

        c = xb.make_computation_builder("pipeline_stage_{}".format(jax_pipeline_stage.name))
        xla_consts = xla._xla_consts(c, consts)
        xla_args, _ = xla._xla_callable_args(c, in_avals, tuple_args, donated_invars=None)
        axis_env = xla.AxisEnv(nreps=1, names=(), sizes=())  # All named axes have been vmapped
        out_nodes = xla.jaxpr_subcomp(
            c, closed_jaxpr.jaxpr, backend, axis_env, xla_consts,
            extend_name_stack(wrap_name(jax_pipeline_stage.name, 'stage')), *xla_args)
        out_tuple = xc.ops.Tuple(c, out_nodes)
        built = c.build(out_tuple)

        return cls(
            name=jax_pipeline_stage.name,
            hlo_proto=built.as_serialized_hlo_module_proto(),
            invars=jax_pipeline_stage.invars,
            pipeline_invars=jax_pipeline_stage.pipeline_invars,
            global_invars=jax_pipeline_stage.global_invars,
            local_invars=jax_pipeline_stage.local_invars,
            outvars=jax_pipeline_stage.outvars,
            pipeline_outvars=jax_pipeline_stage.pipeline_outvars,
            global_outvars=jax_pipeline_stage.global_outvars,
            local_outvars=jax_pipeline_stage.local_outvars,
        )

    def get_runnable(self, mesh=None):
        """Return a callable of the pipeline stage."""
        out_avals = [var.aval for var in self.outvars]
        xla_computation = xc.XlaComputation(self.hlo_proto)
        tuple_args = len(self.invars) > 100  # pass long arg lists as tuple for TPU
        backend = 'gpu'
        backend = xb.get_backend(backend)
        device = backend.get_default_device_assignment(1)[0]
        options = get_compile_options(
            num_replicas=1,
            num_partitions=1,
            device_assignment=(device.id,) if device else None,
            use_spmd_partitioning=False,
            parameter_is_tupled_arguments=tuple_args,
            build_random_seed=42,
        )

        compiled = backend.compile(xla_computation, compile_options=options)
        result_handlers = map(partial(xla.aval_to_result_handler, device), out_avals)
        kept_var_idx = range(len(self.invars))
        return partial(xla._execute_compiled, compiled, out_avals, result_handlers, kept_var_idx)


@dataclass
class StrVarPipelineStage:
    """Stringified stage with all Set/Dict have string keys."""

    name: str
    # invars
    invars: Sequence[str]
    pipeline_invars: Set[str]
    global_invars: Set[str]
    local_invars: Set[str]
    # outvars
    outvars: Sequence[str]
    pipeline_outvars: Set[str]
    global_outvars: Set[str]
    local_outvars: Set[str]

    @classmethod
    def from_pipeline_stage(cls, pipeline_stage: PipelineStage):
        """Construct a StrVarPipelineStage from a PipelineStage."""
        return cls(
            name=pipeline_stage.name,
            invars=[repr(var) for var in pipeline_stage.invars],
            pipeline_invars={repr(var) for var in pipeline_stage.pipeline_invars},
            global_invars={repr(var) for var in pipeline_stage.global_invars},
            local_invars={repr(var) for var in pipeline_stage.local_invars},
            outvars=[repr(var) for var in pipeline_stage.outvars],
            pipeline_outvars={repr(var) for var in pipeline_stage.pipeline_outvars},
            global_outvars={repr(var) for var in pipeline_stage.global_outvars},
            local_outvars={repr(var) for var in pipeline_stage.local_outvars},
        )


@dataclass
class XlaShardedPipelineStage(PipelineStage):
    """Pipeline stage with XLA HLO protos, supporting auto-sharding."""

    mesh: VirtualMesh = None
    hlo_module: Any = None
    hlo_proto: Any = None
    donated_invars: Any = None  # TODO(Hao): figure out donated_invars
    compiled: Any = None
    strategy_config: StrategyConfig = None

    @classmethod
    def from_jax_pipeline_stage(cls,
                                *,
                                jax_pipeline_stage: JaxPipelineStage,
                                mesh: VirtualMesh,
                                donated_invars,
                                memory_budget_per_device):
        # pylint: disable=too-many-locals
        """Run auto-sharding optimizer on a Jax pipeline stage."""
        distributed_compilation_head = bool(mesh.is_distributed)
        logical_mesh = mesh.get_default_logical_mesh()
        closed_jaxpr = jax_pipeline_stage.closed_jaxpr()
        in_avals = [var.aval for var in jax_pipeline_stage.invars]
        consts = closed_jaxpr.consts
        # map(xla.prefetch, it.chain(consts, xla.jaxpr_literals(closed_jaxpr.jaxpr)))

        # Convert jaxpr to XLA HLO
        c = xb.make_computation_builder("pipeline_stage_{}".format(jax_pipeline_stage.name))
        xla_consts = map(partial(xb.constant, c), consts)
        tuple_args = False
        xla_args, _ = xla._xla_callable_args(c, in_avals, tuple_args, donated_invars=None)
        backend_name = "gpu"
        axis_env = xla.AxisEnv(nreps=1, names=(), sizes=())  # All named axes have been vmapped
        out_nodes = xla.jaxpr_subcomp(
            c, closed_jaxpr.jaxpr, backend_name, axis_env, xla_consts,
            extend_name_stack(wrap_name(jax_pipeline_stage.name, "stage")), *xla_args)
        out_tuple = xc.ops.Tuple(c, out_nodes)

        # Set up aliases (donating invars)
        # backend = xb.get_backend(backend_name)
        # Note(Hao): we do not handle donation now.
        # if backend.platform in ("gpu", "tpu"):
        #     # donation_results = xla.set_up_aliases(c, xla_args, out_tuple, donated_invars, tuple_args)
        #     donation_results = xla.set_up_aliases(c, xla_args, out_tuple, donated_invars, tuple_args)
        # if any(donation_results):
        #     # TODO(tomhennigan): At call time we should mark these buffers as deleted.
        #     unused_donations = [str(c.GetShape(a))
        #                         for a, d in zip(xla_args, donation_results) if d]
        #     warn("Some donated buffers were not usable: {}".format(", ".join(unused_donations)))

        # Compile and optimize HLO to an executable
        built = c.Build(out_tuple)
        backend = xb.get_backend(backend_name)
        compiled, strategy_config = compile_with_search(
            backend, built, mesh,
            "cost_model", [logical_mesh], memory_budget_per_device,
            search_task=None, record_file=None)
        hlo_module = compiled.hlo_modules()[0]

        logger.debug(">>> Stage sharding solution vector: {}.".format(
            strategy_config.auto_sharding_solution_vector))

        return cls(
            name=jax_pipeline_stage.name,
            mesh=mesh,
            hlo_module=hlo_module,
            hlo_proto=built.as_serialized_hlo_module_proto(),
            compiled=compiled,
            strategy_config=strategy_config,
            donated_invars=donated_invars,
            invars=jax_pipeline_stage.invars,
            pipeline_invars=jax_pipeline_stage.pipeline_invars,
            global_invars=jax_pipeline_stage.global_invars,
            local_invars=jax_pipeline_stage.local_invars,
            outvars=jax_pipeline_stage.outvars,
            pipeline_outvars=jax_pipeline_stage.pipeline_outvars,
            global_outvars=jax_pipeline_stage.global_outvars,
            local_outvars=jax_pipeline_stage.local_outvars,
        )

    def get_runnable(self, mesh=None):
        """Return a callable of the pipeline stage."""
        if not isinstance(mesh, PhysicalDeviceMesh):
            raise RuntimeError("Require a pre-allocated physical mesh to compile the runnable.")
        assert isinstance(mesh, PhysicalDeviceMesh)

        compiled = self.compiled
        strategy_config = self.strategy_config
        logical_mesh_shape = strategy_config.logical_mesh_shape
        if self.mesh.is_distributed:
            compiled = mesh.compile_remote_executable(self.hlo_proto, strategy_config)

        avals = [var.aval for var in self.invars]
        out_avals = [var.aval for var in self.outvars]
        input_shardings = self.hlo_module.spmd_parameters_shardings()
        input_sharding_specs = [hlo_sharding_to_sharding_spec(proto_tuple, aval, logical_mesh_shape)
                                for (proto_tuple, aval) in zip(input_shardings, avals)]
        output_sharding = self.hlo_module.spmd_output_sharding()
        output_sharding_specs = hlo_sharding_to_sharding_spec(output_sharding, out_avals, logical_mesh_shape)

        # Return the final callable
        return mesh.get_callable_with_arg_handler(compiled, avals, out_avals,
                                                  input_sharding_specs, output_sharding_specs,
                                                  self.donated_invars)

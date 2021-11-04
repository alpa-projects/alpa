"""pipeline computation definitions."""
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
import itertools as it
import logging
from typing import Sequence, List, Set, Any, Dict

import numpy as np
from jax import jit, linear_util as lu
from jax._src.util import partial, safe_map
from jax.core import Atom, Var, JaxprEqn, Jaxpr, ClosedJaxpr, DropVar, Literal, jaxpr_as_fun, new_jaxpr_eqn, gensym
from jax.interpreters import xla
from jax.lax import add_p
from jax.lib import xla_bridge as xb, xla_client as xc
from jax.core import (Atom, ClosedJaxpr, JaxprEqn, Jaxpr, Var, Literal, DropVar,
                      gensym, new_jaxpr_eqn, jaxpr_as_fun)

from parax.device_mesh import PhysicalDeviceMesh
from parax.measure_record import StrategyConfig
from parax.mesh_executable import PartialGradAccMeshDriverExecutable
from parax.pipeline_parallel.primitive_def import (pipeline_p,
                                                   mark_pipeline_jaxpreqn)
from parax.shard_parallel.auto_sharding import (compile_with_search,
                                                compile_with_given_strategy,
                                                get_input_output_sharding_specs,
                                                HloProtoStatus)
from parax.util import get_compile_options, jaxpr_to_hlo_computation, setup_computation_alias

# pylint: disable=redefined-builtin
unsafe_map, map = map, safe_map  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class PipelineComputation(ABC):
    """
    Base class of pipeline computations.

    Attributes:
        name (str): The name of the pipeline computation.
        invars (Sequence[Var]): The list of input variables, corresponding to
            the order of the runnable inputs.
        outvars (Sequence[Var]): The list of output variables, corresponding to
            the order of the runnable outputs.
    """

    name: str
    invars: Sequence[Var] = field(default_factory=list)
    outvars: Sequence[Var] = field(default_factory=list)

    @abstractmethod
    def get_runnable(self, mesh=None):
        """Compile the computation and get the runnable."""
        raise NotImplementedError()


@dataclass
class StrVarPipelineComputation:
    """Stringified computation with all Set/Dict have string keys."""

    name: str
    invars: Sequence[str]
    outvars: Sequence[str]

    @classmethod
    def from_pipeline_computation(cls,
                                  pipeline_computation: PipelineComputation):
        """Construct a StrVarPipelineComputation from a PipelineComputation."""
        return cls(
            name=pipeline_computation.name,
            invars=[repr(var) for var in pipeline_computation.invars],
            outvars=[repr(var) for var in pipeline_computation.outvars],
        )


@dataclass
class JaxPipelineComputation(PipelineComputation):
    """
    A pipeline computation defined by Jaxpr.

    Attributes:
        eqns (List[JaxprEqn]): Jaxpr equations of the pipeline computation.
        consts_dir: Dict[Atom, Any]: All the constants used in the pipeline
            computation.
    """

    eqns: List[JaxprEqn] = field(default_factory=list)
    consts_dir: Dict[Atom, Any] = field(default_factory=dict)

    def closed_jaxpr(self) -> ClosedJaxpr:
        """
        Get the closed Jaxpr of the pipeline computation.

        Returns:
            ClosedJaxpr: The result ClosedJaxpr.
        """
        jaxpr = Jaxpr(
            constvars=list(self.consts_dir.keys()),
            invars=self.invars,
            outvars=self.outvars,
            eqns=self.eqns,
        )
        closed_jaxpr = ClosedJaxpr(jaxpr, list(self.consts_dir.values()))
        return closed_jaxpr

    def get_runnable(self, mesh=None):
        """Return a JIT callable of the pipeline computation."""
        closed_jaxpr = self.closed_jaxpr()
        return jit(jaxpr_as_fun(closed_jaxpr))

    @classmethod
    def from_closed_jaxpr(cls, name, closed_jaxpr: ClosedJaxpr):
        """Construct a JaxPipelineComputation from a Jaxpr."""
        return cls(name=name,
                   invars=closed_jaxpr.jaxpr.invars,
                   outvars=closed_jaxpr.jaxpr.outvars,
                   eqns=closed_jaxpr.eqns,
                   consts_dir={
                       k: v for k, v in zip(closed_jaxpr.jaxpr.constvars,
                                            closed_jaxpr.consts)
                   })


@dataclass
class XlaPipelineComputation(PipelineComputation):
    """A pipeline computation defined by XLA HLO proto."""

    hlo_proto: bytes = field(default_factory=b"")

    @classmethod
    def from_jax_pipeline_computation(
            cls, jax_pipeline_computation: JaxPipelineComputation):
        """
        Construct a XlaPipelineComputation from a JaxPipelineComputation.

        Args:
            jax_pipeline_computation (JaxPipelineComputation): the source JaxPipelineComputation.
        """
        closed_jaxpr = jax_pipeline_computation.closed_jaxpr()
        backend = xb.get_backend("gpu")
        name = "pipeline_computation_{}".format(jax_pipeline_computation.name)
        built = jaxpr_to_hlo_computation(name, closed_jaxpr, None, backend)

        return cls(
            name=jax_pipeline_computation.name,
            hlo_proto=built.as_serialized_hlo_module_proto(),
            invars=jax_pipeline_computation.invars,
            outvars=jax_pipeline_computation.outvars,
        )

    def get_runnable(self, mesh=None):
        """Return a callable of the pipeline computation."""
        out_avals = [var.aval for var in self.outvars]
        xla_computation = xc.XlaComputation(self.hlo_proto)
        tuple_args = len(
            self.invars) > 100  # pass long arg lists as tuple for TPU
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
        result_handlers = map(partial(xla.aval_to_result_handler, device),
                              out_avals)
        kept_var_idx = range(len(self.invars))
        return partial(xla._execute_compiled, compiled, out_avals,
                       result_handlers, kept_var_idx)


@dataclass
class XlaShardedPipelineComputation(PipelineComputation):
    """A pipeline computation defined by XLA HLO proto. The XLA HLO is annotated by sharding spec."""

    hlo_proto: Any = None
    donated_invars: Any = None
    strategy_config: StrategyConfig = None
    input_sharding_specs: Any = None
    output_sharding_specs: Any = None
    output_acc_grad_indices: Sequence[int] = None

    @classmethod
    def from_auto_sharded_computation(
        cls,
        *,
        jax_pipeline_computation: JaxPipelineComputation,
        auto_sharded_hlo_proto: xc.XlaComputation,
        strategy_config: StrategyConfig,
        donated_invars=None,
        acc_grad_outvars=set()):
        # pylint: disable=too-many-locals
        """Run auto-sharding optimizer on a Jax pipeline computation."""
        if not donated_invars:
            donated_invars = (False,) * len(jax_pipeline_computation.invars)

        acc_grad_indices = [
            out_idx
            for out_idx, outvar in enumerate(jax_pipeline_computation.outvars)
            if outvar in acc_grad_outvars
        ]

        return cls(name=jax_pipeline_computation.name,
                   hlo_proto=auto_sharded_hlo_proto,
                   strategy_config=strategy_config,
                   donated_invars=donated_invars,
                   invars=jax_pipeline_computation.invars,
                   outvars=jax_pipeline_computation.outvars,
                   output_acc_grad_indices=acc_grad_indices)

    # @lu.cache
    def get_compiled(self, mesh=None):

        if not isinstance(mesh, PhysicalDeviceMesh):
            raise RuntimeError(
                "Require a pre-allocated physical mesh to compile the runnable."
            )

        strategy_config = self.strategy_config
        logical_mesh_shape = strategy_config.logical_mesh_shape
        xla_computation = xc.XlaComputation(self.hlo_proto)
        setup_computation_alias(xla_computation, self.donated_invars)
        backend_name = 'gpu'
        backend = xb.get_backend(backend_name)
        num_devices = np.prod(logical_mesh_shape)
        rewrite_for_grad_acc = len(self.output_acc_grad_indices) > 0
        compiled = compile_with_given_strategy(
            backend,
            xla_computation,
            self.strategy_config,
            num_devices,
            mesh.is_distributed,
            HloProtoStatus.SHARDING_ANNOTATED,
            rewrite_for_grad_acc=rewrite_for_grad_acc,
            rewrite_grad_acc_indices=self.output_acc_grad_indices)

        avals = [var.aval for var in self.invars]
        out_avals = [var.aval for var in self.outvars]
        input_sharding_specs, output_sharding_specs = get_input_output_sharding_specs(
            compiled.hlo_modules()[0], num_devices, avals, out_avals,
            strategy_config.logical_mesh_shape)
        self.input_sharding_specs = input_sharding_specs
        self.output_sharding_specs = output_sharding_specs
        return compiled

    def get_runnable(self, mesh=None):
        """Return a callable of the pipeline computation."""

        compiled = self.get_compiled(mesh)
        hlo_module = compiled.hlo_modules()[0]

        # Return the final callable
        avals = [var.aval for var in self.invars]
        out_avals = [var.aval for var in self.outvars]
        mesh_executable = PartialGradAccMeshDriverExecutable(
            mesh, compiled, self.strategy_config, avals, out_avals,
            self.donated_invars, self.output_acc_grad_indices)

        return mesh_executable.get_driver_callable()

    def hlo_proto_str(self):
        xla_computation = xc.XlaComputation(self.hlo_proto)
        return xla_computation.as_hlo_text()


def get_var_mapping(mapping, var):
    if isinstance(var, Var) and var in mapping:
        return mapping[var]
    else:
        return var


def slice_eqns_by_pipeline_marks(closed_jaxpr: ClosedJaxpr):
    sliced_eqns = []
    current_computation_eqns = None

    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            assert current_computation_eqns is None, "Defining a pipeline computation inside a pipeline computation is not allowed."
            current_computation_eqns = []
        elif eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'end':
            assert current_computation_eqns is not None, "Ending a pipeline computation before its start."
            sliced_eqns.append(current_computation_eqns)
            current_computation_eqns = None
        else:
            assert current_computation_eqns is not None
            current_computation_eqns.append(eqn)
    assert current_computation_eqns is None
    return sliced_eqns


def add_pipeline_marks_for_sliced_eqns(closed_jaxpr: ClosedJaxpr, sliced_eqns):
    n_layers = len(sliced_eqns)
    layer_pipeline_invars = [set() for _ in range(n_layers)]
    layer_pipeline_outvars = [set() for _ in range(n_layers)]
    var_layer_dict = {}

    for var in closed_jaxpr.jaxpr.invars:
        var_layer_dict[var] = -1

    for i, eqns in enumerate(sliced_eqns):
        for eqn in eqns:
            for var in eqn.invars:
                if (not isinstance(var, Literal) and
                        var not in closed_jaxpr.jaxpr.constvars and
                        var_layer_dict[var] != i):
                    layer_pipeline_invars[i].add(var)
                    if var_layer_dict[var] == -1:
                        var_layer_dict[var] = i
                    layer_pipeline_outvars[var_layer_dict[var]].add(var)
            for var in eqn.outvars:
                if not isinstance(var, DropVar):
                    var_layer_dict[var] = i

    for var in closed_jaxpr.jaxpr.outvars:
        if (not isinstance(var, Literal) and
                var not in closed_jaxpr.jaxpr.constvars and
                var_layer_dict[var] != -1):
            layer_pipeline_outvars[var_layer_dict[var]].add(var)

    gensym_func = gensym([closed_jaxpr.jaxpr])
    var_mapping = {}

    new_eqns = []
    for i, eqns in enumerate(sliced_eqns):
        # pipeline start eqn
        computation_var_mapping = {}

        pipeline_start_invars = []
        pipeline_start_outvars = []
        for var in layer_pipeline_invars[i]:
            new_var = gensym_func(var.aval)
            pipeline_start_invars.append(get_var_mapping(var_mapping, var))
            pipeline_start_outvars.append(new_var)
            computation_var_mapping[var] = new_var
        new_eqns.append(
            mark_pipeline_jaxpreqn(pipeline_start_invars,
                                   pipeline_start_outvars, str(i), 'start'))
        # all other eqns
        for eqn in eqns:
            new_invars = [
                get_var_mapping(computation_var_mapping, var)
                for var in eqn.invars
            ]
            new_eqns.append(
                new_jaxpr_eqn(new_invars, eqn.outvars, eqn.primitive,
                              eqn.params, eqn.source_info))
        # pipeline end eqn
        pipeline_end_invars = []
        pipeline_end_outvars = []
        for var in layer_pipeline_outvars[i]:
            new_var = gensym_func(var.aval)
            pipeline_end_invars.append(
                get_var_mapping(computation_var_mapping, var))
            pipeline_end_outvars.append(new_var)
            var_mapping[var] = new_var
        new_eqns.append(
            mark_pipeline_jaxpreqn(pipeline_end_invars, pipeline_end_outvars,
                                   str(i), 'end'))
    new_jaxpr = Jaxpr(
        closed_jaxpr.jaxpr.constvars,
        closed_jaxpr.jaxpr.invars,
        [
            get_var_mapping(var_mapping, var)
            for var in closed_jaxpr.jaxpr.outvars
        ],
        new_eqns,
    )
    new_closed_jaxpr = ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)
    return new_closed_jaxpr


def slice_closed_jaxpr_by_full_pipeline_marks(
    closed_jaxpr: ClosedJaxpr
) -> Sequence[JaxPipelineComputation]:  # noqa MC0001
    global_consts_dir = dict(
        zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))

    result_computations = []
    current_computation = None

    from parax.pipeline_parallel.manual_layer_slicing import log_jaxpr
    log_jaxpr(closed_jaxpr, "new_jaxpr")

    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            assert current_computation is None, "Defining a pipeline computation inside a pipeline computation is not allowed."
            current_computation = JaxPipelineComputation(
                name=eqn.params['name'])
            for var in eqn.invars:
                if isinstance(var, Literal):
                    pass
                elif var in global_consts_dir:
                    current_computation.consts_dir[var] = global_consts_dir[var]
                else:
                    current_computation.invars.append(var)

        assert current_computation is not None
        current_computation.eqns.append(eqn)

        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'end':
            assert current_computation is not None, "Ending a pipeline computation before its start."
            assert current_computation.name == eqn.params[
                'name'], "Ending a pipeline computation different from its start."
            for var in eqn.outvars:
                current_computation.outvars.append(var)
            result_computations.append(current_computation)
            current_computation = None

    return result_computations


def mark_missing_vars_in_pipeline_marks(
        computations: Sequence[JaxPipelineComputation], global_invars,
        global_outvars):
    gensym_func = gensym(
        [computation.closed_jaxpr().jaxpr for computation in computations])
    var_computation_id = {}
    for var in global_invars:
        if not isinstance(var, Literal):
            var_computation_id[var] = -1

    computation_additional_invars = [set() for _ in computations]
    computation_additional_outvars = [set() for _ in computations]
    for i, computation in enumerate(computations):
        for eqn in computation.eqns:
            for var in eqn.invars:
                if (not isinstance(var, Literal) and
                        var not in computation.consts_dir and
                        var not in computation.invars):
                    source_computation_id = var_computation_id[var]
                    if source_computation_id != i:
                        if (source_computation_id != -1 and var not in
                                computations[source_computation_id].outvars):
                            computation_additional_outvars[
                                source_computation_id].add(var)
                        computation_additional_invars[i].add(var)
            for var in eqn.outvars:
                var_computation_id[var] = i

    for var in global_outvars:
        source_computation_id = var_computation_id[var]
        if source_computation_id != -1 and var not in computations[
                source_computation_id].outvars:
            computation_additional_outvars[source_computation_id].add(var)

    new_computations = []

    for i, computation in enumerate(computations):
        assert computation.eqns[0].primitive is pipeline_p and computation.eqns[
            0].params['mark_type'] == 'start'
        assert computation.eqns[-1].primitive is pipeline_p and computation.eqns[
            -1].params['mark_type'] == 'end'
        new_computation = JaxPipelineComputation(
            computation.name, consts_dir=computation.consts_dir)

        computation_var_mapping = {
            var: gensym_func(var.aval)
            for var in computation_additional_invars[i] |
            computation_additional_outvars[i]
        }
        pipeline_start_invars = list(computation.eqns[0].invars)
        pipeline_start_outvars = [
            get_var_mapping(computation_var_mapping, var)
            for var in computation.eqns[0].outvars
        ]
        new_computation.invars = list(computation.invars)
        for var in computation_additional_invars[i]:
            pipeline_start_invars.append(var)
            pipeline_start_outvars.append(computation_var_mapping[var])
        pipeline_start_invars_without_literal = []
        pipeline_start_outvars_without_literal = []
        for invar, outvar in zip(pipeline_start_invars, pipeline_start_outvars):
            if isinstance(invar, Literal):
                computation_var_mapping[outvar] = invar
            else:
                pipeline_start_invars_without_literal.append(invar)
                pipeline_start_outvars_without_literal.append(outvar)
        new_computation.invars = list(pipeline_start_invars_without_literal)
        new_computation.eqns.append(computation.eqns[0]._replace(
            invars=pipeline_start_invars_without_literal,
            outvars=pipeline_start_outvars_without_literal))

        for eqn in computation.eqns[1:-1]:
            new_computation.eqns.append(
                eqn._replace(invars=[
                    get_var_mapping(computation_var_mapping, var)
                    for var in eqn.invars
                ],
                             outvars=[
                                 get_var_mapping(computation_var_mapping, var)
                                 for var in eqn.outvars
                             ]))

        pipeline_end_invars = [
            get_var_mapping(computation_var_mapping, var)
            for var in computation.eqns[-1].invars
        ]
        pipeline_end_outvars = list(computation.eqns[-1].outvars)
        for var in computation_additional_outvars[i]:
            pipeline_end_invars.append(computation_var_mapping[var])
            pipeline_end_outvars.append(var)
        pipeline_end_invars_without_dropvar = []
        pipeline_end_outvars_without_dropvar = []
        for invar, outvar in zip(pipeline_end_invars, pipeline_end_outvars):
            if not isinstance(outvar, DropVar):
                pipeline_end_invars_without_dropvar.append(invar)
                pipeline_end_outvars_without_dropvar.append(outvar)
        new_computation.outvars = list(pipeline_end_outvars_without_dropvar)
        new_computation.eqns.append(computation.eqns[-1]._replace(
            invars=pipeline_end_invars_without_dropvar,
            outvars=pipeline_end_outvars_without_dropvar))
        new_computations.append(new_computation)

    return new_computations


def rearrange_vars(vars,
                   selected: Sequence[Var],
                   pipe_marker=None,
                   is_input=True):
    """
    Rearrange vars to let those in selected be the first. If the pipe_marker is given,
    rearrange invars and outvars in pipemarker also.
    Args:
        vars (Sequence[Var]): all vars to be rearranged.
        selected (Sequence[Var]): vars selected to be prior.
        pipe_marker (JaxprEqn): pipe marker corresponding to vars
        is_input (bool): the var is input of pipe_marker, if False, it is output
    """
    new_vars = list(selected)
    selected = set(selected)
    for var in vars:
        if var not in selected:
            new_vars.append(var)

    if pipe_marker is None:
        return new_vars

    if is_input:
        new_invars = new_vars
        invar_idx = {v: idx for idx, v in enumerate(pipe_marker.invars)}
        new_outvars = [
            pipe_marker.outvars[invar_idx[var]] for var in new_invars
        ]
    else:
        new_outvars = new_vars
        outvar_idx = {v: idx for idx, v in enumerate(pipe_marker.outvars)}
        new_invars = [
            pipe_marker.invars[outvar_idx[var]] for var in new_outvars
        ]
    new_marker = mark_pipeline_jaxpreqn(new_invars, new_outvars,
                                        pipe_marker.params['name'],
                                        pipe_marker.params['mark_type'])
    return new_vars, new_marker


def generate_sharded_xla_computations(
        name: str, jax_computations: Sequence[JaxPipelineComputation],
        computation_donate_invars, physical_mesh, logical_mesh_choices,
        logical_mesh_search_mode, memory_budget_per_device, acc_grad_outvars,
        search_task, record_file):
    """Generate sharded XLA computations by running the sharding optimizer given JaxPipelineComputations."""
    invars = set()
    outvars = set()
    donation_mapping = dict()
    eqns = []
    consts_dir = {}
    for computation, donation in zip(jax_computations,
                                     computation_donate_invars):
        consts_dir.update(computation.consts_dir)
        # Do not add local invars into the invars
        invars.update([var for var in computation.invars if var not in outvars])
        outvars.update(computation.outvars)
        for idx, var in enumerate(computation.invars):
            if not donation[idx] or var not in invars:
                continue
            donation_mapping[computation.invars[idx]] = computation.outvars[idx]
        eqns += computation.eqns
    invars = rearrange_vars(invars, donation_mapping.keys())
    outvars = rearrange_vars(outvars, donation_mapping.values())
    jaxpr = Jaxpr(
        constvars=list(consts_dir.keys()),
        invars=list(invars),
        outvars=list(outvars),
        eqns=eqns,
    )

    donation_num = len(donation_mapping)
    dummy_donated_invars = (True,) * donation_num + (False,) * (len(invars) -
                                                                donation_num)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts_dir.values())
    backend_name = 'gpu'
    backend = xb.get_backend(backend_name)
    built = jaxpr_to_hlo_computation(name, closed_jaxpr, None, backend)
    computation_protos, strategy_config = compile_with_search(
        backend,
        built,
        invars,
        outvars,
        dummy_donated_invars,
        physical_mesh,
        logical_mesh_choices,
        logical_mesh_search_mode,
        memory_budget_per_device,
        search_task,
        record_file,
        multiple_stages=True,
        grad_acc_num_micro_batches=None,
        bypass_device_assignment_check=physical_mesh.is_distributed)
    computations = [
        XlaShardedPipelineComputation.from_auto_sharded_computation(
            auto_sharded_hlo_proto=proto,
            jax_pipeline_computation=computation,
            strategy_config=strategy_config,
            donated_invars=donate_invars,
            acc_grad_outvars=acc_grad_outvars)
        for computation, proto, donate_invars in zip(
            jax_computations, computation_protos, computation_donate_invars)
    ]
    return computations


def mark_gradvar_to_mesh(invars: Sequence[Var],
                         computations: Sequence[JaxPipelineComputation],
                         computation_to_mesh, mask):
    # TODO(yonghao): now assume all gradients are variables(not literal)
    outvar2mesh = {}
    for i, computation in enumerate(computations):
        for var in computation.outvars:
            if isinstance(var, Var):
                outvar2mesh[var] = computation_to_mesh[i]
    return {
        invar: outvar2mesh[mask[invar]]
        for invar in invars
        if invar in mask and mask[invar] in outvar2mesh
    }


def get_gradient(compute_jaxpr: ClosedJaxpr, slices: Sequence[ClosedJaxpr]):
    gradients = [
        outvar for outvar in compute_jaxpr.jaxpr.outvars
        if isinstance(outvar, Var)
    ]
    gradients = set(gradients)
    is_gradients = [[
        isinstance(outvar, Var) and outvar in gradients
        for outvar in slice.jaxpr.outvars
    ]
                    for slice in slices]
    return is_gradients


def compute_grad_to_accumulate_grad(compute_jaxpr: ClosedJaxpr, gensym_fn):
    """
    Transform compute grad jaxpr with pipeline markers into accumulate grad jaxpr
    Args:
        compute_jaxpr (ClosedJaxpr): the original jaxpr
        gensym_fn: gensym function
    Returns:
        acc_grad_jaxpr (ClosedJaxpr): The accumulate grad jaxpr
        update_outs (Dict[Var, Var]): From original output(grad) to new output(acc grad)
        grad_in_to_out (Dict[Var, Var]): From accumulated gradient inputs to outputs
    """
    raw_gradients = set([
        outvar for outvar in compute_jaxpr.jaxpr.outvars
        if isinstance(outvar, Var)
    ])
    # Currently, assume no grad is literal
    assert len(raw_gradients) == len(compute_jaxpr.jaxpr.outvars)
    # from raw_gradients to gradients(cross pipeline marker)
    gradients = {}
    reverse_gradients = {}
    for eqn in reversed(compute_jaxpr.eqns):
        if eqn.primitive is pipeline_p:
            for i, outvar in enumerate(eqn.outvars):
                if outvar in raw_gradients:
                    gradients[outvar] = eqn.invars[i]
                    reverse_gradients[eqn.invars[i]] = outvar
                elif outvar in reverse_gradients:
                    # in case that:
                    #   invar = compute gradient
                    #   invar' = pipeline end(invar)
                    #   outvar = pipeline start(invar')
                    #   final = pipeline end(outvar)
                    # gradients[final] should finally maps invar instead of
                    # outvar, then acc grad there
                    final_outvar = reverse_gradients[outvar]
                    gradients[final_outvar] = eqn.invars[i]
                    reverse_gradients[eqn.invars[i]] = final_outvar
    # FIXME(zhuohan): Should support auxiliary outputs in the future (e.g. loss)
    for outvar in raw_gradients:
        assert outvar in gradients, 'all gradients should be captured by pipeline marker'
    grad_values = list(gradients.values())
    # generate new variables
    grad_invars = {outvar: gensym_fn(outvar.aval) for outvar in grad_values}
    grad_outs = {outvar: gensym_fn(outvar.aval) for outvar in grad_values}
    # modify output, here all grads are acc_grad
    new_glob_outvars = []
    new_glob_invars = compute_jaxpr.jaxpr.invars + []
    update_outs = dict()
    grad_in_to_out = dict()
    for outvar in compute_jaxpr.jaxpr.outvars:
        if isinstance(outvar, Var):
            assert outvar in gradients
            new_glob_outvars.append(grad_outs[gradients[outvar]])
            new_glob_invars.append(grad_invars[gradients[outvar]])
            update_outs[outvar] = grad_outs[gradients[outvar]]
            grad_in_to_out[grad_invars[gradients[outvar]]] = grad_outs[
                gradients[outvar]]
        else:
            raise NotImplemented('gradients cannot be Literal')
    gradients = set(grad_values)
    # rewrite eqns
    new_eqns = []
    pipe_start = None
    pipe_eqns = []
    to_acc = []
    for eqn in compute_jaxpr.eqns:
        if eqn.primitive is pipeline_p:
            if eqn.params['mark_type'] == 'start':
                pipe_start = eqn
                for outvar in eqn.outvars:
                    if not isinstance(outvar, DropVar) and outvar in gradients:
                        # collect gradients in this computation
                        to_acc.append(outvar)
                continue
            if eqn.params['mark_type'] == 'end':
                # add grad used in this computation in pipeline start
                grad_in_after_pipe = {
                    outvar: gensym_fn(outvar.aval) for outvar in to_acc
                }
                grad_out_before_pipe = {
                    outvar: gensym_fn(outvar.aval) for outvar in to_acc
                }
                new_pipe_start = mark_pipeline_jaxpreqn(
                    pipe_start.invars + map(lambda x: grad_invars[x], to_acc),
                    pipe_start.outvars +
                    map(lambda x: grad_in_after_pipe[x], to_acc),
                    pipe_start.params['name'], pipe_start.params['mark_type'])
                new_eqns.append(new_pipe_start)
                # add normal eqns
                new_eqns.extend(pipe_eqns)
                # add acc grad(adds)
                for gradient in to_acc:
                    new_eqns.append(
                        new_jaxpr_eqn([grad_in_after_pipe[gradient], gradient],
                                      [grad_out_before_pipe[gradient]], add_p,
                                      {}))
                # add grad created in this computation in pipeline end
                new_pipe_end = mark_pipeline_jaxpreqn(
                    eqn.invars + map(lambda x: grad_out_before_pipe[x], to_acc),
                    eqn.outvars + map(lambda x: grad_outs[x], to_acc),
                    eqn.params['name'], eqn.params['mark_type'])
                new_eqns.append(new_pipe_end)
                pipe_start = None
                pipe_eqns = []
                to_acc = []
                continue
        pipe_eqns.append(eqn)
        for outvar in eqn.outvars:
            if not isinstance(outvar, DropVar) and outvar in gradients:
                # collect gradients in this computation
                to_acc.append(outvar)
    jaxpr = Jaxpr(compute_jaxpr.jaxpr.constvars, new_glob_invars,
                  new_glob_outvars, new_eqns)
    new_closed_jaxpr = ClosedJaxpr(jaxpr, compute_jaxpr.consts)
    # We do not modify donate_invars here, as it is only to append Trues
    # Instead return grad outs to help modify apply_grad
    return new_closed_jaxpr, update_outs, grad_in_to_out


def replace_all_with(closed_jaxpr: ClosedJaxpr, mapping):
    """replace with given mapping of variables"""

    def map_var(var):
        return get_var_mapping(mapping, var)

    new_glob_invars = [map_var(var) for var in closed_jaxpr.jaxpr.invars]
    new_glob_outvars = [map_var(var) for var in closed_jaxpr.jaxpr.outvars]
    new_eqns = []
    for eqn in closed_jaxpr.eqns:
        new_invars = [map_var(var) for var in eqn.invars]
        new_outvars = [map_var(var) for var in eqn.outvars]
        new_eqns.append(
            new_jaxpr_eqn(new_invars, new_outvars, eqn.primitive, eqn.params))
    new_jaxpr = Jaxpr(closed_jaxpr.jaxpr.constvars, new_glob_invars,
                      new_glob_outvars, new_eqns)
    return ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)


def pipeline_dce(jax_pipeline_computations: Sequence[JaxPipelineComputation],
                 global_outvars):
    """
    clear unused vars cross pipeline computations.
    mainly to remove grad and only keep accumulated grad
    """

    def dce_pipe_marker(marker: JaxprEqn, used_set):
        kept_indices = [
            i for i, var in enumerate(marker.outvars) if var in used_set
        ]
        new_marker = mark_pipeline_jaxpreqn(
            [marker.invars[i] for i in kept_indices],
            [marker.outvars[i] for i in kept_indices], marker.params['name'],
            marker.params['mark_type'])
        return new_marker

    global_used = set(global_outvars)
    new_computations = []
    for computation in reversed(jax_pipeline_computations):
        new_eqns = []
        # handle pipe end
        pipe_end = computation.eqns[-1]
        assert (pipe_end.primitive is pipeline_p and
                pipe_end.params['mark_type']
                == 'end'), 'computation not ended by a pipeline marker'
        new_pipe_end = dce_pipe_marker(pipe_end, global_used)
        new_eqns.append(new_pipe_end)
        # handle normal instructions
        local_used = set(new_pipe_end.invars)
        for eqn in reversed(computation.eqns[1:-1]):
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar) and outvar in local_used:
                    new_eqns.append(eqn)
                    local_used.update([
                        invar for invar in eqn.invars if isinstance(invar, Var)
                    ])
        # handle pipe start
        pipe_start = computation.eqns[0]
        assert (pipe_start.primitive is pipeline_p and
                pipe_start.params['mark_type']
                == 'start'), 'computation not started by a pipeline marker'
        new_pipe_start = dce_pipe_marker(pipe_start, local_used)
        new_eqns.append(new_pipe_start)
        global_used.update(new_pipe_start.invars)

        new_eqns = list(reversed(new_eqns))
        new_computation = JaxPipelineComputation(
            computation.name,
            invars=new_pipe_start.invars,
            outvars=new_pipe_end.outvars,
            eqns=new_eqns,
            consts_dir=computation.consts_dir)
        new_computations.append(new_computation)
    new_computations = list(reversed(new_computations))
    return new_computations


def apply_grad_get_mean(closed_jaxpr, gradients, gensym_fn, num_microbatch,
                        global_outvars):
    """
    Get mean of input (accumulated) gradients, run apply gradient
    If the input is output, after this transform it outputs the divided version.
    """
    from jax.lax import div_p
    mapping = dict()
    new_eqns = []
    invar_set = set(closed_jaxpr.jaxpr.invars)
    outvar_set = set(closed_jaxpr.jaxpr.outvars)
    for invar in gradients:
        div_out = gensym_fn(invar.aval)
        new_eqns.append(
            new_jaxpr_eqn(
                [invar,
                 Literal(np.array(num_microbatch, invar.aval.dtype))],
                [div_out], div_p, {}))
        mapping[invar] = div_out
    replaced = replace_all_with(closed_jaxpr, mapping)
    final_invars = closed_jaxpr.jaxpr.invars
    final_outvars = replaced.jaxpr.outvars
    for invar in gradients:
        if invar not in invar_set:
            final_invars.append(invar)
        if invar in global_outvars and invar not in outvar_set:
            final_outvars.append(mapping[invar])
    new_eqns.extend(replaced.jaxpr.eqns)
    new_jaxpr = Jaxpr(closed_jaxpr.jaxpr.constvars, final_invars, final_outvars,
                      new_eqns)
    global_outvars = list(
        map(lambda x: get_var_mapping(mapping, x), global_outvars))
    return ClosedJaxpr(new_jaxpr, closed_jaxpr.consts), global_outvars


def slice_apply_gradient(closed_jaxpr: ClosedJaxpr, grad_mesh: Dict[Var, int],
                         mesh_num):
    """
    Slice the apply gradient jaxpr based on mesh allocation information
    Args:
        closed_jaxpr (ClosedJaxpr): closed jaxpr of apply_gradient function.
        grad_mesh (Dict[Var, int]): dict indicating which mesh the variable is at.
        If not in the dict, the variable should be a global parameter.
        mesh_num (int): number of meshes. If a mesh does not have apply gradient computation,
        add an empty jaxpr
    Returns:
        jaxprs(List[ClosedJaxpr]): The i-th ClosedJaxpr runs at the i-th cluster.
        info: A tuple of:
            deps (List[Tuple[int, int]]): Indicating dependencies of apply gradient computations
            mesh_assignment (Dict[int, int]): Indicating mesh the apply grad computation is assigned
            infered_global_invars (Dict[Var, List[int]]): Indicating which clusters each
            input variable of apply_gradient function should be sent to.
    """

    def add_allocation(cur: Set, add: Set):
        if cur is None:
            return add
        else:
            return cur.union(add)

    global_invars = closed_jaxpr.jaxpr.invars
    eqn_mesh = dict()
    var_mesh = {var: set([mesh]) for var, mesh in grad_mesh.items()}
    infered_global_invars = dict()
    constvars = [list() for _ in range(mesh_num)]
    consts = [list() for _ in range(mesh_num)]
    sliced_eqns = [list() for _ in range(mesh_num)]
    invars = [list() for _ in range(mesh_num)]
    outvars = [list() for _ in range(mesh_num)]
    # propagate mesh assignments from input
    for eqn_idx, eqn in enumerate(closed_jaxpr.eqns):
        at_mesh = None
        for invar in eqn.invars:
            if isinstance(invar, Var) and invar in var_mesh:
                at_mesh = add_allocation(at_mesh, var_mesh[invar])
        if at_mesh is not None:
            for invar in eqn.invars:
                if isinstance(invar, Var):
                    cur_mesh = var_mesh[invar] if invar in var_mesh else None
                    var_mesh[invar] = add_allocation(cur_mesh, at_mesh)
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar):
                    var_mesh[outvar] = at_mesh
            eqn_mesh[eqn_idx] = at_mesh
    # propagate back
    for reversed_idx, eqn in enumerate(reversed(closed_jaxpr.eqns)):
        eqn_idx = len(closed_jaxpr.eqns) - 1 - reversed_idx
        if eqn_idx not in eqn_mesh:
            at_mesh = None
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar) and outvar in var_mesh:
                    at_mesh = add_allocation(at_mesh, var_mesh[outvar])
            if at_mesh is not None:
                eqn_mesh[eqn_idx] = at_mesh
                for invar in eqn.invars:
                    if isinstance(invar, Var):
                        cur_mesh = var_mesh[invar] if invar in var_mesh else None
                        var_mesh[invar] = add_allocation(cur_mesh, at_mesh)
    for eqn_idx, eqn in enumerate(closed_jaxpr.eqns):
        if eqn_idx in eqn_mesh:
            for mesh in eqn_mesh[eqn_idx]:
                sliced_eqns[mesh].append(eqn)
        else:
            # all inputs are infered, all outputs are not assigned
            sliced_eqns[0].append(eqn)
            for invar in eqn.invars:
                if isinstance(invar, Var):
                    if invar not in var_mesh:
                        var_mesh[invar] = [0]
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar):
                    assert (outvar not in var_mesh or
                            (len(var_mesh[outvar]) == 1 and
                             var_mesh[outvar][0] == 0))
                    var_mesh[outvar] = [0]
    # grouping invars and outvars
    for invar in global_invars:
        assert invar in var_mesh
        for mesh in var_mesh[invar]:
            invars[mesh].append(invar)
        infered_global_invars[invar] = var_mesh[invar]
    for outvar in closed_jaxpr.jaxpr.outvars:
        assert outvar in var_mesh
        for mesh in var_mesh[outvar]:
            outvars[mesh].append(outvar)
    # grouping consts and constvars
    for aval, var in zip(closed_jaxpr.consts, closed_jaxpr.jaxpr.constvars):
        assert var in var_mesh
        for mesh in var_mesh[var]:
            consts[mesh].append(aval)
            constvars[mesh].append(var)

    jaxprs = []
    deps = []
    mesh_assignment = {}

    for i in range(mesh_num):
        if not outvars[i]:
            continue
        computation_idx = mesh_num * 2 + len(jaxprs)
        # assign the current computation into mesh i
        mesh_assignment[computation_idx] = i
        for v in invars[i]:
            if v in grad_mesh:
                # Add dependency as (computation, compute grad computation)
                deps.append((computation_idx, mesh_num * 2 - 1 - grad_mesh[v]))
        jaxprs.append(
            ClosedJaxpr(
                Jaxpr(constvars[i], invars[i], outvars[i], sliced_eqns[i]),
                consts[i]))

    info = deps, mesh_assignment, infered_global_invars
    return jaxprs, info


def apply_grad_add_marker(jaxprs, mask, gensym_fn, computation=False):
    """
    Add pipeline markers for sliced apply grads, keep invars and outvars still unless
    the invar is in mask or invar is outvar.
    In the first case, the final invar follows the mask;
    In the second case, the final outvar is recorded in outvar_map
    Args:
        jaxprs(Sequence[ClosedJaxpr]): sliced apply grads.
        mask: mask[gradient] is the corresponding accumulated gradient(real invar).
        gensym_fn: gensym function of the whole jaxpr.
        computation(Bool): output JaxPipelineComputation or ClosedJaxpr.
    """
    results = []
    outvar_map = dict()
    for i, jaxpr in enumerate(jaxprs):
        new_map = dict()
        for invar in jaxpr.jaxpr.invars:
            if invar not in mask:
                new_map[invar] = gensym_fn(invar.aval)
        for outvar in jaxpr.jaxpr.outvars:
            if not isinstance(outvar, Var):
                raise NotImplemented("outvar of apply grad cannot be literal")
            if outvar in jaxpr.jaxpr.invars:
                if outvar not in outvar_map:
                    outvar_map[outvar] = gensym_fn(outvar.aval)
                continue
            new_map[outvar] = gensym_fn(outvar.aval)
        replaced = replace_all_with(jaxpr, new_map).jaxpr
        new_invars = list(
            map(lambda x: get_var_mapping(mask, x), jaxpr.jaxpr.invars))
        new_outvars = list(
            map(lambda x: get_var_mapping(outvar_map, x), jaxpr.jaxpr.outvars))
        name = str(i) + '_apply_grad'
        start_marker = mark_pipeline_jaxpreqn(new_invars,
                                              replaced.invars,
                                              name=name,
                                              mark_type='start')
        end_marker = mark_pipeline_jaxpreqn(replaced.outvars,
                                            new_outvars,
                                            name=name,
                                            mark_type='end')
        new_eqns = [start_marker] + replaced.eqns + [end_marker]
        if computation:
            results.append(
                JaxPipelineComputation(
                    name, new_invars, new_outvars, new_eqns,
                    dict(zip(jaxpr.jaxpr.constvars, jaxpr.consts))))
        else:
            new_jaxpr = Jaxpr(jaxpr.jaxpr.constvars, new_invars, new_outvars,
                              new_eqns)
            results.append(ClosedJaxpr(new_jaxpr, jaxpr.consts))
    return results, outvar_map


def merge_computation_jaxprs(jaxprs: Sequence[ClosedJaxpr],
                             used: Set[Var],
                             new_marker_name,
                             donation_mapping=None) -> ClosedJaxpr:
    """
    Merge continuous jaxprs and remove pipe markers
    Args:
        jaxprs (Sequence[ClosedJaxpr]): jaxprs to be merged
        used (Set[Var]): out variables used later
        new_marker_name (str): name of merged pipeline used in marker
        donation_mapping (Dict[Var, Var]): donation mapping of merged jaxpr, may have redundant items
    """
    new_invars = dict()
    new_outvars = dict()
    new_eqns = []
    var_map = dict()

    # handle const vars:
    new_constvars = dict()
    for jaxpr in jaxprs:
        new_constvars.update(dict(zip(jaxpr.jaxpr.constvars, jaxpr.consts)))

    for idx, jaxpr in enumerate(jaxprs):
        # handle pipeline start marker:
        pipe_start = jaxpr.eqns[0]
        for invar, outvar in zip(pipe_start.invars, pipe_start.outvars):
            if invar not in var_map:
                # is not local output, the outvar is kept
                if invar in new_constvars:
                    continue
                # is already set in earlier computations
                if invar in new_invars:
                    var_map[outvar] = new_invars[invar]
                    continue
                new_invars[invar] = outvar
            else:
                # is local output, the outvar is redirected
                var_map[outvar] = var_map[invar]
        # handle normal eqns
        for eqn in jaxpr.eqns[1:-1]:
            new_local_invars = [get_var_mapping(var_map, v) for v in eqn.invars]
            new_eqns.append(
                new_jaxpr_eqn(new_local_invars, eqn.outvars, eqn.primitive,
                              eqn.params, eqn.source_info))
        # handle pipeline end marker
        pipe_end = jaxpr.eqns[-1]
        for invar, outvar in zip(pipe_end.invars, pipe_end.outvars):
            if outvar in used:
                new_outvars[outvar] = get_var_mapping(var_map, invar)
            var_map[outvar] = get_var_mapping(var_map, invar)

    new_pipe_start = mark_pipeline_jaxpreqn(list(new_invars.keys()),
                                            list(new_invars.values()),
                                            new_marker_name, 'start')
    new_pipe_end = mark_pipeline_jaxpreqn(list(new_outvars.values()),
                                          list(new_outvars.keys()),
                                          new_marker_name, 'end')
    new_eqns = [new_pipe_start] + new_eqns + [new_pipe_end]
    constvars = set(new_constvars.keys())
    new_invars = [k for k in new_invars.keys() if k not in constvars]
    new_outvars = list(new_outvars.keys())
    if donation_mapping:
        new_invars_set = set(new_invars)
        new_outvars_set = set(new_outvars)
        donation_mapping = {
            k: v
            for k, v in donation_mapping.items()
            if k in new_invars_set and v in new_outvars_set
        }
        new_invars = rearrange_vars(new_invars, donation_mapping.keys())
        new_outvars = rearrange_vars(new_outvars, donation_mapping.values())
    return ClosedJaxpr(
        Jaxpr(list(new_constvars.keys()), new_invars, new_outvars, new_eqns),
        list(new_constvars.values()))

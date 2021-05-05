from dataclasses import dataclass, field
from typing import List, Set, Any, Dict
from jax.core import Atom, Var, JaxprEqn, Jaxpr, ClosedJaxpr


@dataclass
class PipelineStage:
    name: str
    eqns: List[JaxprEqn] = field(default_factory=list)
    consts_dir: Dict[Atom, Any] = field(default_factory=dict)
    # invars
    pipeline_invars: Set[Var] = field(default_factory=set)
    global_invars: Set[Var] = field(default_factory=set)
    local_invars: Set[Var] = field(default_factory=set)
    # outvars
    pipeline_outvars: Set[Var] = field(default_factory=set)
    global_outvars: Set[Var] = field(default_factory=set)
    local_outvars: Set[Var] = field(default_factory=set)
    # intermediate vars
    intermediate_vars: Set[Var] = field(default_factory=set)

    def closed_jaxpr(self):
        jaxpr = Jaxpr(
            constvars=self.consts_dir.keys(),
            invars=list(self.pipeline_invars | self.global_invars | self.local_invars),
            outvars=list(self.pipeline_outvars | self.global_outvars | self.local_outvars),
            eqns=self.eqns,
        )
        closed_jaxpr = ClosedJaxpr(jaxpr, self.consts_dir.values())
        return closed_jaxpr


class PicklableStage(PipelineStage):
    """Ray cannot pickle xla_extension.traceback, or custom JVP."""
    @staticmethod
    def from_pipeline_stage(stage):
        pickabled_stage = PicklableStage(name=stage.name,
                                         consts_dir=stage.consts_dir,
                                         pipeline_invars=stage.pipeline_invars,
                                         global_invars=stage.global_invars,
                                         local_invars=stage.local_invars,
                                         pipeline_outvars=stage.pipeline_outvars,
                                         global_outvars=stage.global_outvars,
                                         local_outvars=stage.local_outvars,
                                         intermediate_vars=stage.intermediate_vars)
        # Now re-process source_info: XLA_extension.Traceback in eqns
        # TODO(Hao): serialize and deserialize it to preserve it.
        # Now I assume it has maximally two layers, but we can make it recursive.
        eqns = []
        for eqn in stage.eqns:
            new_eqn = JaxprEqn(invars=eqn.invars,
                               outvars=eqn.outvars,
                               primitive=eqn.primitive,
                               params=eqn.params,
                               source_info=None)
            for param, val in new_eqn.params.items():
                if type(val) == ClosedJaxpr:
                    for i, eq in enumerate(val.jaxpr.eqns):
                        new_eq = JaxprEqn(invars=eq.invars,
                                          outvars=eq.outvars,
                                          primitive=eq.primitive,
                                          params=eq.params,
                                          source_info=None)
                        val.jaxpr.eqns[i] = new_eq

            eqns.append(new_eqn)
        pickabled_stage.eqns = eqns
        return pickabled_stage

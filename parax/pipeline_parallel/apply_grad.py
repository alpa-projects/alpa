"""Transformations and utilities to process gradient accumulation and apply_gradient."""
from abc import ABC, abstractmethod
import logging
from typing import Sequence, Dict

from jax._src.util import safe_map
from jax.core import Var, Jaxpr, ClosedJaxpr, DropVar, Literal, new_jaxpr_eqn
from jax.lax import add_p
import numpy as np

from parax.pipeline_parallel.computation import JaxPipelineComputation
from parax.pipeline_parallel.layer_construction import get_var_mapping
from parax.pipeline_parallel.primitive_def import (pipeline_p,
                                                   mark_pipeline_jaxpreqn)
from parax.pipeline_parallel.schedules import gen_dependency_with_stages
from parax.util import slices_to_jaxpr, OrderedSet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

unsafe_map, map = map, safe_map  # type: ignore


def split_compute_grad_and_apply_grad(closed_jaxpr: ClosedJaxpr):
    """Split the train_step jaxpr into two parts: compute_grad and apply_grad."""
    split_eqn = None
    for idx, eqn in enumerate(closed_jaxpr.eqns):
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'grad':
            split_eqn = eqn
            split_idx = idx
    if split_eqn is None:
        logger.warning(
            "Missing barrier between compute and apply. Assume there is no "
            "apply gradient step. Hint: replace jax.grad by parax.grad.")
        return closed_jaxpr, ClosedJaxpr(Jaxpr([], [], [], []), []), None
    sliced_eqns = [
        closed_jaxpr.eqns[:split_idx], [split_eqn],
        closed_jaxpr.eqns[split_idx + 1:]
    ]
    compute_grad, _, apply_grad = slices_to_jaxpr(closed_jaxpr, sliced_eqns)
    if len(apply_grad.eqns) == 0:
        logger.warning(
            "the apply gradient part is empty. Hint: apply() after parax.grad")
    return compute_grad, apply_grad, split_eqn


def compute_grad_to_accumulate_grad(compute_jaxpr: ClosedJaxpr, gensym_fn):
    """
    Transform compute_grad jaxpr with pipeline markers into accumulate_grad jaxpr.

    Args:
        compute_jaxpr (ClosedJaxpr): the original jaxpr
        gensym_fn: gensym function
    Returns:
        acc_grad_jaxpr (ClosedJaxpr): The accumulate grad jaxpr
        update_outs (Dict[Var, Var]): From original output(grad) to new output(acc grad)
        grad_in_to_out (Dict[Var, Var]): From accumulated gradient inputs to outputs
    """
    raw_gradients = OrderedSet([
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
        assert outvar in gradients, "all gradients should be captured by pipeline marker"
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
            raise NotImplemented("gradients cannot be Literal")
    gradients = OrderedSet(grad_values)
    # rewrite eqns
    new_eqns = []
    pipe_start = None
    last_pipe_end = None
    pipe_eqns = []
    to_acc = []
    for eqn in compute_jaxpr.eqns:
        if eqn.primitive is pipeline_p:
            if eqn.params["mark_type"] == "start":
                pipe_start = eqn
                for outvar in eqn.outvars:
                    if not isinstance(outvar, DropVar) and outvar in gradients:
                        # collect gradients in this computation
                        to_acc.append(outvar)
                continue
            if eqn.params["mark_type"] == "end":
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
                last_pipe_end = new_pipe_end
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
    last_pipe_end.params["name"] += "_grad_acc_boundary"
    jaxpr = Jaxpr(compute_jaxpr.jaxpr.constvars, new_glob_invars,
                  new_glob_outvars, new_eqns)
    new_closed_jaxpr = ClosedJaxpr(jaxpr, compute_jaxpr.consts)
    # We do not modify donate_invars here, as it is only to append Trues
    # Instead return grad outs to help modify apply_grad
    return new_closed_jaxpr, update_outs, grad_in_to_out


def get_apply_grad_outvar_constraints(jax_pipeline_stages, stage_to_mesh,
                                      global_invars, donated_invars,
                                      donation_mapping):
    outvar_mesh = dict()
    donated_global_vars = set([
        invar for invar, donate in zip(global_invars, donated_invars) if donate
    ])
    for stage_idx, stage in enumerate(jax_pipeline_stages):
        for invar in stage.invars:
            if invar in donated_global_vars:
                outvar_mesh.setdefault(donation_mapping[invar],
                                       OrderedSet()).add(
                                           stage_to_mesh[stage_idx])
    return outvar_mesh


def process_apply_gradient(apply_grad_jaxpr, barrier, acc_grad_dict,
                           jax_pipeline_stages, stage_to_mesh, gensym_func,
                           num_micro_batches, num_meshes, global_invars,
                           global_outvars, donated_invars):
    """Slice apply_grad jaxpr into stages and assign them to the correspondig meshes."""
    # TODO(yonghao): the condition of creating RDA variable should be extended.

    # Process apply gradient:
    # 1. change invars of apply grad to outvars of accumulate grad
    gradients = [g for g in barrier.outvars if not isinstance(g, DropVar)]
    mask = {
        outv: acc_grad_dict[inv] for outv, inv in zip(gradients, barrier.invars)
    }

    # 2. Add compute mean and slice apply-grad stages
    gradvar_to_mesh = get_var_to_mesh(gradients, jax_pipeline_stages,
                                      stage_to_mesh, mask)
    # FIXME (zhuohan): get_mean only works when we use jax.mean to
    #                  calculate loss. It will fail if we use sum.
    apply_grad_jaxpr, global_outvars = apply_grad_get_mean(
        apply_grad_jaxpr, gradients, gensym_func, num_micro_batches,
        global_outvars)

    # update donation mapping
    donation_mapping = dict()
    for idx, invar in enumerate(global_invars):
        if donated_invars[idx]:
            donation_mapping[invar] = global_outvars[idx]
    # create outvar constraints
    outvar_mesh = get_apply_grad_outvar_constraints(jax_pipeline_stages,
                                                    stage_to_mesh,
                                                    global_invars,
                                                    donated_invars,
                                                    donation_mapping)

    sliced_apply_grad, info = slice_apply_gradient(apply_grad_jaxpr,
                                                   gradvar_to_mesh, outvar_mesh,
                                                   num_meshes, donation_mapping)
    apply_deps, apply_grad_placement, _ = info
    sliced_apply_grad, out_map = apply_grad_add_marker(sliced_apply_grad,
                                                       mask,
                                                       gensym_func,
                                                       computation=True)
    global_outvars = list(
        map(lambda x: get_var_mapping(out_map, x), global_outvars))
    n_stages = len(jax_pipeline_stages) + len(sliced_apply_grad)
    dependency = gen_dependency_with_stages(jax_pipeline_stages,
                                            len(sliced_apply_grad), apply_deps)

    used_simultaneously = OrderedSet()
    used = OrderedSet()
    for stage in sliced_apply_grad:
        used_simultaneously.update(used.intersection(stage.invars))
        used.update(stage.invars)

    donated_invars = list(donated_invars)
    for idx, invar in enumerate(global_invars):
        if invar in used_simultaneously and donated_invars[idx]:
            if len(invar.aval.shape) != 0:
                logger.warning(
                    f"Cannot donate {invar} (shape: {invar.aval.shape})")
            donated_invars[idx] = False

    return (sliced_apply_grad, n_stages, dependency, apply_grad_placement,
            global_outvars, donated_invars)


def replace_all_with(closed_jaxpr: ClosedJaxpr, mapping):
    """Replace all variables in a jaxpr given the mapping."""

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


def apply_grad_get_mean(closed_jaxpr, gradients, gensym_fn, num_microbatch,
                        global_outvars):
    """
    Get mean of input (accumulated) gradients, run apply gradient
    If the input is output, after this transform it outputs the divided version.
    """
    from jax.lax import div_p
    mapping = dict()
    new_eqns = []
    invar_set = OrderedSet(closed_jaxpr.jaxpr.invars)
    outvar_set = OrderedSet(closed_jaxpr.jaxpr.outvars)
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
                         outvar_mesh: Dict[Var, OrderedSet[int]], mesh_num,
                         donation_mapping):
    """
    Slice the apply gradient jaxpr based on mesh allocation information
    Args:
        closed_jaxpr (ClosedJaxpr): closed jaxpr of apply_gradient function.
        grad_mesh (Dict[Var, int]): some invars should be at certain mesh.
        If not in the dict, the variable should be a global parameter.
        outvar_mesh (Dict[Var, int]): some outvars should be at certain mesh
        mesh_num (int): number of meshes. If a mesh does not have apply gradient computation,
        add an empty jaxpr
        donation_mapping (Dict[Var, Var]): donation mapping for global invars
    Returns:
        jaxprs(List[ClosedJaxpr]): The i-th ClosedJaxpr runs at the i-th cluster.
        info: A tuple of:
            deps (List[Tuple[int, int]]): dependencies of apply gradient computations
            mesh_assignment (Dict[int, int]): From apply grad index to the its mesh's index
            infered_global_invars (Dict[Var, List[int]]): From invar index to meshes need
            this invar.
    """

    global_invars = closed_jaxpr.jaxpr.invars
    eqn_mesh = dict()
    var_mesh = {var: OrderedSet([mesh]) for var, mesh in grad_mesh.items()}
    infered_global_invars = dict()
    constvars = [list() for _ in range(mesh_num)]
    consts = [list() for _ in range(mesh_num)]
    sliced_eqns = [list() for _ in range(mesh_num)]
    invars = [list() for _ in range(mesh_num)]
    outvars = [list() for _ in range(mesh_num)]
    for var in outvar_mesh:
        var_mesh.setdefault(var, OrderedSet()).update(outvar_mesh[var])
    # propagate mesh assignments from input
    for eqn_idx, eqn in enumerate(closed_jaxpr.eqns):
        at_mesh = OrderedSet()
        for invar in eqn.invars:
            if isinstance(invar, Var):
                at_mesh.update(var_mesh.setdefault(invar, OrderedSet()))
        if at_mesh:
            for invar in eqn.invars:
                if isinstance(invar, Var):
                    cur_mesh = var_mesh.setdefault(invar, OrderedSet())
                    cur_mesh.update(at_mesh)
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar):
                    var_mesh[outvar] = OrderedSet(at_mesh)
            eqn_mesh[eqn_idx] = OrderedSet(at_mesh)
    changed = True
    while changed:
        changed = False
        # propagate back
        for reversed_idx, eqn in enumerate(reversed(closed_jaxpr.eqns)):
            eqn_idx = len(closed_jaxpr.eqns) - 1 - reversed_idx
            origin_at_mesh: OrderedSet = eqn_mesh.setdefault(
                eqn_idx, OrderedSet())
            at_mesh = OrderedSet()
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar):
                    at_mesh.update(var_mesh.setdefault(outvar, OrderedSet()))
            if not at_mesh:
                continue
            if (not origin_at_mesh or at_mesh.difference(origin_at_mesh)):
                changed = True
                origin_at_mesh.update(at_mesh)
                for invar in eqn.invars:
                    if isinstance(invar, Var):
                        var_mesh.setdefault(invar, OrderedSet()).update(at_mesh)
        for invar in closed_jaxpr.jaxpr.invars:
            if invar in donation_mapping:
                outvar = donation_mapping[invar]
                outvar_at = var_mesh.setdefault(outvar, OrderedSet())
                invar_at = var_mesh.setdefault(invar, OrderedSet())
                if invar_at.difference(outvar_at):
                    outvar_at.update(invar_at)
                    changed = True
                if outvar_at.difference(invar_at):
                    invar_at.update(outvar_at)

    for eqn_idx, eqn in enumerate(closed_jaxpr.eqns):
        if eqn_mesh[eqn_idx]:
            assert len(eqn_mesh[eqn_idx])
            for mesh in eqn_mesh[eqn_idx]:
                sliced_eqns[mesh].append(eqn)
        else:
            # all inputs are infered, all outputs are not assigned
            sliced_eqns[0].append(eqn)
            logger.debug(f'{eqn} are arbitrarily assigned')
            for invar in eqn.invars:
                if isinstance(invar, Var):
                    if not var_mesh.setdefault(invar, OrderedSet()):
                        var_mesh[invar].add(0)
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar):
                    assert (not var_mesh.setdefault(outvar, OrderedSet()) or
                            (len(var_mesh[outvar]) == 1 and
                             var_mesh[outvar][0] == 0))
                    var_mesh[outvar].add(0)

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


def get_var_to_mesh(invars: Sequence[Var],
                    computations: Sequence[JaxPipelineComputation],
                    computation_to_mesh, mask):
    """Get the mapping from variables to mesh"""
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

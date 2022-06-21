"""Transformations and utilities to process gradient accumulation and
apply_gradient."""
import logging
from typing import Sequence, Dict, Tuple

from jax._src.util import safe_map
from jax.core import (Primitive, Var, Jaxpr, ClosedJaxpr, DropVar, Literal,
                      get_aval, raise_to_shaped, JaxprEqn)
from jax.interpreters import xla
from jax.lax import add_p, div_p, and_p, or_p
from jaxlib import xla_client as xc
import numpy as np

from alpa.pipeline_parallel.computation import JaxPipelineComputation
from alpa.pipeline_parallel.primitive_def import (pipeline_p,
                                                  mark_pipeline_jaxpreqn)
from alpa.pipeline_parallel.schedules import gen_dependency_with_stages
from alpa.util import (clone_jaxpr, clone_jaxpr_eqn, slices_to_jaxpr,
                       OrderedSet, get_var_mapping, new_jaxpr_eqn)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# pylint: disable=redefined-builtin
unsafe_map, map = map, safe_map  # type: ignore
APPLY_GRAD_MARKER_SUFFIX = 'apply_grad'


def _value_to_literal(value, dtype):
    literal_val = np.array(value, dtype)
    return Literal(literal_val, raise_to_shaped(get_aval(literal_val)))


# TODO(yonghao): delaying the cross layer grad accmulation increases memory
# cost, but may not decrease communication: if c=a+b is delayed, both a and
# b are accumulated, so the memory cost is more than when only accumulate c.
# If layer that outputs a(called layer_a, and the same applys for b) is
# merged with layer_b to the same stage, they do not need any communication,
# so the communication does not benefit from the rewrite.
def _rewrite_cross_layer_grad(compute_eqns, microbatch_bound, apply_eqns,
                              gensym_fn, closed_jaxpr):
    """
    If a parameter is used in multiple stages, its gradient is computed in
    multiple stages and then added together. We accumulate the results on each
    stage, and add them together exactly at the start of apply grad period.

    A common use case is the tied embedding in language models.
    """
    unmarked_vars = set()
    layer_invars = set()
    global_invars = closed_jaxpr.jaxpr.invars
    for eqn in compute_eqns:
        if eqn.primitive is pipeline_p:
            if eqn.params['mark_type'] == 'end':
                unmarked_vars.update(
                    [v for v in eqn.outvars if not isinstance(v, DropVar)])
            elif eqn.params['mark_type'] == 'start':
                layer_invars.update(
                    [v for v in eqn.invars if isinstance(v, Var)])
    cross_layer_grad_eqns = []
    new_compute_eqns = []
    # Those eqn directly use output of pipeline end is delayed to apply grad.
    defined_vars = set(global_invars)
    for eqn in compute_eqns:
        if eqn.primitive is pipeline_p:
            new_compute_eqns.append(eqn)
            continue
        invars = [v for v in eqn.invars if isinstance(v, Var)]
        if not unmarked_vars.intersection(invars):
            new_compute_eqns.append(eqn)
            continue
        assert unmarked_vars.issuperset(invars), f"'{eqn}' is not fully marked."
        outvars = [v for v in eqn.outvars if not isinstance(v, DropVar)]
        assert not layer_invars.intersection(
            outvars), f"'{eqn}' cannot be delayed."
        cross_layer_grad_eqns.append(eqn)
        unmarked_vars.update(outvars)
        defined_vars.update(outvars)
    # Rewrite microbatch_bound and cross_layer_grad eqns.
    microbatch_bound_in_to_outs = {}
    for invar, outvar in zip(microbatch_bound.invars, microbatch_bound.outvars):
        if isinstance(invar, Var) and not isinstance(outvar, DropVar):
            microbatch_bound_in_to_outs[invar] = outvar
    new_cross_microbatch_bound_invars = OrderedSet()
    new_post_microbatch_bound_outvars = OrderedSet()
    for eqn in cross_layer_grad_eqns:
        for invar in eqn.invars:
            if (isinstance(invar, Var) and
                    invar not in microbatch_bound_in_to_outs and
                    invar not in defined_vars):
                new_cross_microbatch_bound_invars.add(invar)
                microbatch_bound_in_to_outs[invar] = gensym_fn(invar.aval)
        new_post_microbatch_bound_outvars.update([
            var for var in eqn.outvars if not isinstance(var, DropVar) and
            var in microbatch_bound_in_to_outs
        ])
    # rewrite the microbatch_bound
    new_microbatch_bound_invars = []
    new_microbatch_bound_outvars = []
    for idx, var in enumerate(microbatch_bound.invars +
                              list(new_cross_microbatch_bound_invars)):
        # remove vars now defined after microbatch_bound.
        if isinstance(var, Var) and var in new_post_microbatch_bound_outvars:
            continue
        new_microbatch_bound_invars.append(var)
        # add vars now used after microbatch_bound.
        new_microbatch_bound_outvars.append(
            microbatch_bound.outvars[idx] if idx < len(microbatch_bound.invars)
            else microbatch_bound_in_to_outs[var])
    new_microbatch_bound = clone_jaxpr_eqn(microbatch_bound,
                                           new_microbatch_bound_invars,
                                           new_microbatch_bound_outvars)
    # rewrite cross layer grad eqns and insert them to the top of apply eqns.
    new_apply_eqns = []
    rewrite_invars = set(new_microbatch_bound_invars)
    rewrite_invars.update(microbatch_bound.invars)
    for eqn in cross_layer_grad_eqns:
        invars = [
            microbatch_bound_in_to_outs[var]
            if isinstance(var, Var) and var in rewrite_invars else var
            for var in eqn.invars
        ]
        outvars = [
            microbatch_bound_in_to_outs[var]
            if not isinstance(var, DropVar) and var in rewrite_invars else var
            for var in eqn.outvars
        ]
        new_apply_eqns.append(clone_jaxpr_eqn(eqn, invars, outvars))
    new_apply_eqns += apply_eqns
    new_global_outvars = list(closed_jaxpr.jaxpr.outvars)
    for idx in range(len(new_global_outvars)):
        var = new_global_outvars[idx]
        if isinstance(var, Literal):
            continue
        if var in rewrite_invars:
            new_global_outvars[idx] = microbatch_bound_in_to_outs[var]
    closed_jaxpr = clone_jaxpr(closed_jaxpr,
                               eqns=new_compute_eqns + [new_microbatch_bound] +
                               new_apply_eqns,
                               outvars=new_global_outvars)
    return closed_jaxpr, [
        new_compute_eqns, [new_microbatch_bound], new_apply_eqns
    ]


def jaxpr_have_apply_grad(closed_jaxpr: ClosedJaxpr):
    """Returns True if the jaxpr has apply_grad."""
    return any(eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'grad'
               for eqn in closed_jaxpr.eqns)


def split_compute_grad_and_apply_grad(closed_jaxpr: ClosedJaxpr, gensym_fn,
                                      num_microbatch: int,
                                      inference_mode: bool):
    """Split the train_step jaxpr into two parts: compute_grad and
    apply_grad. These two parts are separated by a gradient marker generated
    by `alpa.grad`."""
    split_eqn = None
    for idx, eqn in enumerate(closed_jaxpr.eqns):
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'grad':
            split_eqn = eqn
            split_idx = idx
    if split_eqn is None:
        if not inference_mode:
            logger.warning(
                'Missing microbatch_bound between compute and apply. '
                'Assume there is no apply gradient step. '
                'Hint: replace jax.grad by alpa.grad.')
        dummy_jaxpr = ClosedJaxpr(Jaxpr([], [], [], []), [])
        invars = list(closed_jaxpr.jaxpr.outvars) if num_microbatch > 1 else []
        outvars = list(closed_jaxpr.jaxpr.outvars) if num_microbatch > 1 else []
        dummy_bound = new_jaxpr_eqn(invars, outvars, pipeline_p, {
            'mark_type': 'grad',
            'name': ''
        })
        return closed_jaxpr, closed_jaxpr, dummy_jaxpr, dummy_bound
    sliced_eqns = [
        closed_jaxpr.eqns[:split_idx], split_eqn,
        closed_jaxpr.eqns[split_idx + 1:]
    ]
    closed_jaxpr, sliced_eqns = _rewrite_cross_layer_grad(
        *sliced_eqns, gensym_fn, closed_jaxpr)
    sliced_jaxprs = slices_to_jaxpr(closed_jaxpr, sliced_eqns)
    compute_grad, _, apply_grad = sliced_jaxprs  # pylint: disable=unbalanced-tuple-unpacking
    split_eqn = sliced_eqns[1][0]
    if len(apply_grad.eqns) == 0:
        logger.warning(
            'the apply gradient part is empty. Hint: apply() after alpa.grad')
    return closed_jaxpr, compute_grad, apply_grad, split_eqn


def _get_post_to_pre_marker_mapping(compute_jaxpr):
    """
    Get a dict that maps an out_var of a pipeline marker to
    its corresponding in_var.
    """
    post_marker_outs = [
        outvar for outvar in compute_jaxpr.jaxpr.outvars
        if isinstance(outvar, Var)
    ]
    # Currently, assume no grad is literal
    assert len(post_marker_outs) == len(compute_jaxpr.jaxpr.outvars)
    post_marker_outs = OrderedSet(post_marker_outs)
    # from post_marker_outs to post_to_pre_marker_outs(cross pipeline marker)
    post_to_pre_marker_outs = {}
    pre_to_post_marker_outs = {}
    for eqn in reversed(compute_jaxpr.eqns):
        if eqn.primitive is pipeline_p:
            for i, outvar in enumerate(eqn.outvars):
                if outvar in post_marker_outs:
                    post_to_pre_marker_outs[outvar] = eqn.invars[i]
                    pre_to_post_marker_outs[eqn.invars[i]] = outvar
                elif outvar in pre_to_post_marker_outs:
                    # in case that:
                    #   invar = compute gradient
                    #   invar' = pipeline end(invar)
                    #   outvar = pipeline start(invar')
                    #   final = pipeline end(outvar)
                    # post_to_pre_marker_outs[final] = invar' instead of outvar
                    final_outvar = pre_to_post_marker_outs[outvar]
                    post_to_pre_marker_outs[final_outvar] = eqn.invars[i]
                    pre_to_post_marker_outs[eqn.invars[i]] = final_outvar
    for outvar in post_marker_outs:
        assert outvar in post_to_pre_marker_outs, (
            'all outputs should be captured by pipeline marker')
    return post_to_pre_marker_outs


def _rewrite_jaxpr_to_reduced_outputs(compute_jaxpr, to_reduce_pre_marker_outs,
                                      reduce_invars, reduce_outvars, gensym_fn):
    new_eqns = []
    pipe_start = None
    pipe_eqns = []
    to_acc = []
    to_reduce_pre_marker_outs = OrderedSet(to_reduce_pre_marker_outs)
    for eqn in compute_jaxpr.eqns:
        if eqn.primitive is pipeline_p:
            if eqn.params['mark_type'] == 'start':
                pipe_start = eqn
                for outvar in eqn.outvars:
                    if (not isinstance(outvar, DropVar) and
                            outvar in to_reduce_pre_marker_outs):
                        # collect to_reduce_pre_marker_outs in this computation
                        to_acc.append(outvar)
                continue
            if eqn.params['mark_type'] == 'end':
                # add grad used in this computation in pipeline start
                reduce_invar_post_pipe = {
                    outvar: gensym_fn(outvar.aval) for outvar in to_acc
                }
                reduce_outvar_pre_pipe = {
                    outvar: gensym_fn(outvar.aval) for outvar in to_acc
                }
                new_pipe_start = mark_pipeline_jaxpreqn(
                    pipe_start.invars + map(lambda x: reduce_invars[x], to_acc),
                    pipe_start.outvars +
                    # pylint: disable=cell-var-from-loop
                    map(lambda x: reduce_invar_post_pipe[x], to_acc),
                    pipe_start.params['name'],
                    pipe_start.params['mark_type'])
                new_eqns.append(new_pipe_start)
                # add normal eqns
                new_eqns.extend(pipe_eqns)
                # add acc grad(adds)
                for gradient in to_acc:
                    new_eqns.append(
                        new_jaxpr_eqn(
                            [reduce_invar_post_pipe[gradient], gradient],
                            [reduce_outvar_pre_pipe[gradient]], add_p, {}))
                # add grad created in this computation in pipeline end
                new_pipe_end = mark_pipeline_jaxpreqn(
                    # pylint: disable=cell-var-from-loop
                    eqn.invars +
                    map(lambda x: reduce_outvar_pre_pipe[x], to_acc),
                    eqn.outvars + map(lambda x: reduce_outvars[x], to_acc),
                    eqn.params['name'],
                    eqn.params['mark_type'])
                new_eqns.append(new_pipe_end)
                pipe_start = None
                pipe_eqns = []
                to_acc = []
                continue
        pipe_eqns.append(eqn)
        for outvar in eqn.outvars:
            if (not isinstance(outvar, DropVar) and
                    outvar in to_reduce_pre_marker_outs):
                # collect to_reduce_pre_marker_outs in this computation
                to_acc.append(outvar)
    return new_eqns


# TODO(yonghao): support not only reduction and concate. Some outputs may not
# rely on batch dimension.
def compute_grad_to_accumulate_grad(
        compute_jaxpr: ClosedJaxpr, microbatch_bound: JaxprEqn,
        reduction_vector: Sequence[bool], gensym_fn,
        num_microbatch) -> Tuple[ClosedJaxpr, JaxprEqn, Dict[Var, Var]]:
    """Transform compute_grad jaxpr with pipeline markers into accumulate_grad
    jaxpr.

    Args:
        compute_jaxpr: the original jaxpr
        microbatch_bound: The boundary eqn that separates compute_grad and
          apply_grad.
        reduction_vector: if the outvar is reduced(accumulated) or not
        gensym_fn: gensym function

    Returns:
        acc_grad_jaxpr: The accumulate grad jaxpr
        microbatch_bound: The updated microbatch boundary
        reduced_in_to_out: From accumulated gradient inputs to outputs
    """
    if num_microbatch <= 1:
        return compute_jaxpr, microbatch_bound, {}

    post_to_pre_marker_outs = _get_post_to_pre_marker_mapping(compute_jaxpr)
    to_reduce_pre_marker_outs = []
    for var, reduced in zip(compute_jaxpr.jaxpr.outvars, reduction_vector):
        if reduced:
            to_reduce_pre_marker_outs.append(post_to_pre_marker_outs[var])
    # generate new variables
    reduced_invars = {
        outvar: gensym_fn(outvar.aval) for outvar in to_reduce_pre_marker_outs
    }
    reduced_outvars = {
        outvar: gensym_fn(outvar.aval) for outvar in to_reduce_pre_marker_outs
    }
    # modify output, here all grads are acc_grad
    new_glob_outvars = []
    new_glob_invars = compute_jaxpr.jaxpr.invars + []
    update_outs = {}
    reduced_in_to_out = {}
    for outvar, reduced in zip(compute_jaxpr.jaxpr.outvars, reduction_vector):
        if not reduced:
            new_glob_outvars.append(outvar)
            update_outs[outvar] = outvar
        elif isinstance(outvar, Var):
            assert outvar in post_to_pre_marker_outs
            pre_marker_outvar = post_to_pre_marker_outs[outvar]
            reduced_outvar = reduced_outvars[pre_marker_outvar]
            reduced_invar = reduced_invars[pre_marker_outvar]

            new_glob_outvars.append(reduced_outvar)
            new_glob_invars.append(reduced_invar)
            update_outs[outvar] = reduced_outvar
            reduced_in_to_out[reduced_invar] = reduced_outvar
        else:
            raise NotImplementedError('outputs cannot be Literal')
    # rewrite eqns
    new_eqns = _rewrite_jaxpr_to_reduced_outputs(compute_jaxpr,
                                                 to_reduce_pre_marker_outs,
                                                 reduced_invars,
                                                 reduced_outvars, gensym_fn)

    new_closed_jaxpr = clone_jaxpr(compute_jaxpr, new_glob_invars,
                                   new_glob_outvars, new_eqns)

    microbatch_bound_invars = [update_outs[x] for x in microbatch_bound.invars]
    microbatch_bound = clone_jaxpr_eqn(microbatch_bound,
                                       microbatch_bound_invars)
    return new_closed_jaxpr, microbatch_bound, reduced_in_to_out


def _get_apply_grad_outvar_constraints(pipeline_stages, stage_to_mesh,
                                       global_invars, donated_invars,
                                       donation_mapping):
    """Infer outvar constraints of apply gradient based on donation."""
    outvar_mesh = {}
    donated_global_vars = {
        invar for invar, donate in zip(global_invars, donated_invars) if donate
    }
    for stage_idx, stage in enumerate(pipeline_stages):
        for invar in stage.invars:
            if invar in donated_global_vars:
                outvar_mesh.setdefault(donation_mapping[invar],
                                       OrderedSet()).add(
                                           stage_to_mesh[stage_idx])
    return outvar_mesh


def process_apply_gradient(apply_grad_jaxpr, microbatch_bound, pipeline_stages,
                           stage_to_mesh, gensym_func, num_micro_batches,
                           num_meshes, global_invars, global_outvars,
                           donated_invars, reduction_vector, profiling,
                           mesh_num_devices):
    """Slice apply_grad jaxpr into stages and assign them to the corresponding
    meshes."""
    # Process apply gradient:
    # 1. change invars of apply grad to outvars of accumulate grad
    gradients = [
        g for g in microbatch_bound.outvars if not isinstance(g, DropVar)
    ]
    assert len(gradients) == len(microbatch_bound.invars)
    apply_in_to_acc_out = dict(zip(gradients, microbatch_bound.invars))

    # 2. Add compute mean and slice apply-grad stages
    gradvar_to_mesh = get_var_to_mesh(gradients, pipeline_stages, stage_to_mesh,
                                      apply_in_to_acc_out)
    # FIXME (zhuohan): get_mean only works when we use jax.mean to
    #                  calculate loss. It will fail if we use sum.
    apply_grad_jaxpr, global_outvars = apply_grad_get_mean(
        apply_grad_jaxpr, global_outvars, gradients, gensym_func,
        num_micro_batches, reduction_vector)

    # update donation mapping
    donation_mapping = {}
    for idx, invar in enumerate(global_invars):
        if donated_invars[idx]:
            donation_mapping[invar] = global_outvars[idx]
    # create outvar constraints
    outvar_mesh = _get_apply_grad_outvar_constraints(pipeline_stages,
                                                     stage_to_mesh,
                                                     global_invars,
                                                     donated_invars,
                                                     donation_mapping)

    sliced_apply_grad, info = slice_apply_gradient(
        apply_grad_jaxpr, gradvar_to_mesh, outvar_mesh, num_meshes,
        len(pipeline_stages), donation_mapping, gensym_func, profiling,
        mesh_num_devices)
    apply_grad_placement, _, allreduce_groups = info
    sliced_apply_grad, out_map = apply_grad_add_marker(sliced_apply_grad,
                                                       apply_in_to_acc_out,
                                                       gensym_func,
                                                       computation=True)
    global_outvars = list(
        map(lambda x: get_var_mapping(out_map, x), global_outvars))
    n_stages = len(pipeline_stages) + len(sliced_apply_grad)
    dependency = gen_dependency_with_stages(pipeline_stages, sliced_apply_grad)

    return (sliced_apply_grad, n_stages, dependency, apply_grad_placement,
            global_outvars, donated_invars, allreduce_groups)


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
        new_eqns.append(clone_jaxpr_eqn(eqn, new_invars, new_outvars))
    new_jaxpr = clone_jaxpr(closed_jaxpr, new_glob_invars, new_glob_outvars,
                            new_eqns)
    return new_jaxpr


def apply_grad_get_mean(closed_jaxpr, global_outvars, gradients, gensym_fn,
                        num_microbatch, reduce_invars):
    """
    Get the mean of input (accumulated) gradients and run apply gradient.

    If the input is output, after this transform it outputs the divided version.
    """
    mapping = {}
    new_eqns = []
    invar_set = OrderedSet(closed_jaxpr.jaxpr.invars)
    outvar_set = OrderedSet(closed_jaxpr.jaxpr.outvars)
    for invar, reduce in zip(gradients, reduce_invars):
        if not reduce:
            mapping[invar] = invar
            continue
        div_out = gensym_fn(invar.aval)
        new_eqns.append(
            new_jaxpr_eqn([
                invar,
                _value_to_literal(num_microbatch, invar.aval.dtype),
            ], [div_out], div_p, {}))
        mapping[invar] = div_out
    replaced = replace_all_with(closed_jaxpr, mapping)
    final_invars = list(closed_jaxpr.jaxpr.invars)
    final_outvars = list(replaced.jaxpr.outvars)
    for invar, reduce in zip(gradients, reduce_invars):
        if not reduce:
            continue
        if invar not in invar_set:
            final_invars.append(invar)
        if invar in global_outvars and invar not in outvar_set:
            # use the divided version to replace the original one
            final_outvars.append(mapping[invar])
    new_eqns.extend(replaced.jaxpr.eqns)
    new_jaxpr = Jaxpr(closed_jaxpr.jaxpr.constvars, final_invars, final_outvars,
                      new_eqns)
    new_jaxpr = clone_jaxpr(closed_jaxpr, final_invars, final_outvars, new_eqns)
    global_outvars = list(
        map(lambda x: get_var_mapping(mapping, x), global_outvars))
    return new_jaxpr, global_outvars


cross_mesh_allreduce_p = Primitive('__builtin$CrossMeshAllReduce')
_primitive_to_str = {add_p: b'SUM', and_p: b'AND', or_p: b'OR'}


def _cross_mesh_allreduce_xla_translation(c, *args, **kwargs):
    call_name = b'__builtin$CrossMeshAllReduce'
    assert len(args) == 1
    input_params = args[0]
    input_shape = c.get_shape(input_params)
    op_type = _primitive_to_str[kwargs['type']]

    # TODO(yonghao): the has_side_effect is to prevent CSE of the allreduce.
    # It might be replaced by adding its outvar to output
    output = xc.ops.CustomCall(c,
                               call_name,
                               operands=(input_params,),
                               shape=input_shape,
                               has_side_effect=True,
                               opaque=op_type)
    c.clear_sharding()
    return output


xla.translations[cross_mesh_allreduce_p] = _cross_mesh_allreduce_xla_translation


def _propagate_var_at_mesh(eqns, var_mesh):
    """Propagate mesh assignments from input."""
    eqn_mesh = {}
    var_mesh = dict(var_mesh)
    allreduce_outvars = set()
    for eqn_idx, eqn in enumerate(eqns):
        at_mesh = OrderedSet()
        at_each_mesh = True
        for invar in eqn.invars:
            if isinstance(invar, Var) and not invar in allreduce_outvars:
                at_mesh.update(var_mesh.setdefault(invar, OrderedSet()))
                at_each_mesh = False
        if at_mesh:
            eqn_mesh[eqn_idx] = OrderedSet(at_mesh)
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar):
                    var_mesh[outvar] = OrderedSet(at_mesh)
            if eqn.primitive == cross_mesh_allreduce_p:
                allreduce_outvars.add(eqn.outvars[0])
                continue
            for invar in eqn.invars:
                if isinstance(invar, Var):
                    cur_mesh = var_mesh.setdefault(invar, OrderedSet())
                    cur_mesh.update(at_mesh)
        elif at_each_mesh:
            # This var is the result of jaxprs created from all vars
            allreduce_outvars.update(
                [v for v in eqn.outvars if not isinstance(v, DropVar)])
    return eqn_mesh, var_mesh


def _reverse_propagate_var_at_mesh(closed_jaxpr, donation_mapping, eqn_mesh,
                                   var_mesh):
    """Propagate var_at_mesh from output to make sure all operands are ready."""
    changed = False
    for reversed_idx, eqn in enumerate(reversed(closed_jaxpr.eqns)):
        if eqn.primitive == cross_mesh_allreduce_p:
            continue
        eqn_idx = len(closed_jaxpr.eqns) - 1 - reversed_idx
        post_at_mesh = eqn_mesh.setdefault(eqn_idx, OrderedSet())
        at_mesh = OrderedSet()
        for outvar in eqn.outvars:
            if not isinstance(outvar, DropVar):
                at_mesh.update(var_mesh.setdefault(outvar, OrderedSet()))
        if not at_mesh:
            continue
        if (not post_at_mesh or at_mesh.difference(post_at_mesh)):
            changed = True
            post_at_mesh.update(at_mesh)
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
    return changed


def _apply_grad_group_vars(closed_jaxpr: ClosedJaxpr, var_mesh, num_mesh):
    """Slice the input, output and consts of the jaxpr based on var_mesh."""
    global_invars = closed_jaxpr.jaxpr.invars
    invars = [[] for _ in range(num_mesh)]
    outvars = [[] for _ in range(num_mesh)]
    constvars = [[] for _ in range(num_mesh)]
    consts = [[] for _ in range(num_mesh)]
    infered_global_invars = {}
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
    return (invars, outvars, consts, constvars), infered_global_invars


# Binary operators that satisfies the associativity and commutativity
_reducable_operators = set([add_p, and_p, or_p])


class ApplyGradRewriter:
    """
    Rewrite apply grad jaxpr to avoid replicated computation by inserting
    cross-mesh allreduce.
    """

    def __init__(self, apply_grad_jaxpr: ClosedJaxpr, var_mesh):
        self.jaxpr = apply_grad_jaxpr
        self.eqns = apply_grad_jaxpr.jaxpr.eqns
        self.outvars = apply_grad_jaxpr.jaxpr.outvars
        self.var_mesh = dict(var_mesh)
        self.eqn_mesh = {}
        self.var_use: Dict[Var, OrderedSet] = {}
        self.var_def: Dict[Var, int] = {}

    def _reducable(self, eqn):
        # the is_scalar is to avoid a large all-reduce for tied-embedding
        # it can be improved by adding computation-communication tradeoff
        return (eqn.primitive in _reducable_operators and
                eqn.outvars[0].aval.shape == ())

    def _var_at_one_mesh(self, var):
        if not isinstance(var, Var):
            return True
        return var in self.var_mesh and len(self.var_mesh[var]) == 1

    def _other_invar_at_one_mesh(self, var, dst):
        op = dst.invars[0] if var == dst.invars[1] else dst.invars[1]
        return self._var_at_one_mesh(op)

    def _forward_propagate(self):
        """
        A conservative propagation that stops when the eqn's invars are from
        multiple meshes.
        """
        self.eqn_mesh = {}
        var_mesh = dict(self.var_mesh)
        self.var_use = {}
        self.var_def = {}
        # Propagate the first round
        for eqn_idx, eqn in enumerate(self.eqns):
            at_mesh = OrderedSet()
            for invar in eqn.invars:
                if isinstance(invar, Var):
                    at_mesh.update(var_mesh.setdefault(invar, OrderedSet()))
                    self.var_use.setdefault(invar, OrderedSet()).add(eqn_idx)
            if len(at_mesh) == 1:
                for invar in eqn.invars:
                    if isinstance(invar, Var):
                        var_mesh.setdefault(invar, OrderedSet()).update(at_mesh)
                self.eqn_mesh[eqn_idx] = list(at_mesh)[0]
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar):
                    var_mesh[outvar] = OrderedSet(at_mesh)
                    self.var_def[outvar] = eqn_idx
        for eqn_idx, eqn in enumerate(self.eqns):
            if eqn_idx in self.eqn_mesh:
                mesh = self.eqn_mesh[eqn_idx]
                for var in eqn.invars:
                    if isinstance(var, Var):
                        self.var_mesh[var] = OrderedSet([mesh])
                for var in eqn.outvars:
                    if not isinstance(var, DropVar):
                        self.var_mesh[var] = OrderedSet([mesh])

    def _reducable_chain_lookup(self, eqn_idx, num_mesh):
        """
        Pattern matching. For: c = a + b and e = c + d, if a, b and d are all at
        one mesh, and c is only used once, then we can do the reduction to
        translate additions into an allreduce.
        """
        # List[mesh_idx -> List[Vars]]
        mesh_vars = [[] for _ in range(num_mesh)]
        literals = []
        eqn = self.eqns[eqn_idx]
        cur_var = eqn.invars[0]
        nxt_idx, nxt_eqn = eqn_idx, eqn
        reducable_chain = []
        while (self._reducable(nxt_eqn) and
               (nxt_eqn.primitive == eqn.primitive) and
               self._other_invar_at_one_mesh(cur_var, nxt_eqn)):
            cur_idx, cur_eqn = nxt_idx, nxt_eqn
            cur_var = cur_eqn.outvars[0]
            reducable_chain.append(cur_idx)
            outv_use = self.var_use.setdefault(cur_eqn.outvars[0], OrderedSet())
            if len(outv_use) != 1 or cur_eqn.outvars[0] in self.outvars:
                break
            nxt_idx = list(outv_use)[0]
            nxt_eqn = self.eqns[nxt_idx]
        final_var = cur_eqn.outvars[0]
        # split eqns on the reducable chain into meshes
        reducable_set = set(reducable_chain)
        for reduced_idx in reducable_chain:
            reduced_eqn = self.eqns[reduced_idx]
            for op in reduced_eqn.invars:
                if isinstance(op, Literal):
                    mesh_vars[0].append(op)
                    continue
                def_idx = self.var_def[op]
                if def_idx not in reducable_set:
                    mesh_vars[self.eqn_mesh[def_idx]].append(op)
        return mesh_vars, final_var, reducable_chain[:-1], literals

    def _rewrite_eqns(self, primitive, mesh_vars, gensym_fn, outvar, literals):
        # rewrite according to splits
        # TODO: in some cases the literal can lead to final result(True&or_p)
        appended_eqns = []
        allreduce_vars = []
        mesh_ids = []
        literal_handled = False
        for mesh_id, per_mesh_vars in enumerate(mesh_vars):
            cur_val = None
            for v in per_mesh_vars:
                if cur_val is None:
                    cur_val = v
                    continue
                new_var = gensym_fn(cur_val.aval)
                appended_eqns.append(
                    new_jaxpr_eqn([cur_val, v], [new_var], primitive, {}))
                cur_val = new_var
            if cur_val is not None:
                if not literal_handled:
                    for literal in literals:
                        new_var = gensym_fn(cur_val.aval)
                        appended_eqns.append(
                            new_jaxpr_eqn([cur_val, literal], [new_var],
                                          primitive, {}))
                        cur_val = new_var
                    literal_handled = True
                allreduce_vars.append(cur_val)
                mesh_ids.append(mesh_id)
        # modify the end of reduce chain eqn into an all-reduce.
        # The allreduce will be immediately replaced by pipeline markers
        appended_eqns.append(
            new_jaxpr_eqn(allreduce_vars, [outvar], cross_mesh_allreduce_p,
                          {'type': primitive}))
        return appended_eqns, mesh_ids

    def split_replicated_eqns(self, gensym_fn, num_mesh):
        """Rewrite apply grad jaxpr to eqns so as to """
        self._forward_propagate()
        new_eqns_before_var = {}
        # Try to match the pattern
        removed_eqns = set()
        allreduce_groups = OrderedSet()
        for eqn_idx, eqn in enumerate(self.eqns):
            # Do not handle c = a(mesh1 and 2) + b(mesh1) case
            if (eqn_idx not in self.eqn_mesh and self._reducable(eqn) and
                    self._var_at_one_mesh(eqn.invars[0]) and
                    self._var_at_one_mesh(eqn.invars[1])):
                (mesh_vars, final_var, removed,
                 literals) = self._reducable_chain_lookup(eqn_idx, num_mesh)
                removed_eqns.update(removed)
                appended_eqns, allreduce_group = self._rewrite_eqns(
                    eqn.primitive, mesh_vars, gensym_fn, final_var, literals)
                new_eqns_before_var[final_var] = appended_eqns
                allreduce_groups.add(tuple(allreduce_group))
        if len(allreduce_groups) > 1:
            raise NotImplementedError()
        new_eqns = []
        for eqn_idx, eqn in enumerate(self.eqns):
            if eqn_idx in removed_eqns:
                continue
            outv = eqn.outvars[0] if len(eqn.outvars) > 0 else None
            # insert new eqns before the previous last available eqn
            if (not (outv is None or isinstance(outv, DropVar)) and
                    outv in new_eqns_before_var):
                new_eqns.extend(new_eqns_before_var[outv])
            else:
                new_eqns.append(eqn)
        return clone_jaxpr(self.jaxpr, eqns=new_eqns), tuple(allreduce_groups)

    @staticmethod
    def rewrite_allreduce(closed_jaxpr: ClosedJaxpr, rewrite_to_dummy,
                          num_devices, gensym_fn):
        """For cross-mesh allreduce, rewrite its invar to make it legal."""
        vars = set()
        new_eqns = []
        vars.update([
            inv for inv in closed_jaxpr.jaxpr.invars
            if not isinstance(inv, Var)
        ])
        for eqn in closed_jaxpr.eqns:
            if eqn.primitive == cross_mesh_allreduce_p:
                new_invars = set(eqn.invars).intersection(vars)
                assert len(new_invars) == 1
                if rewrite_to_dummy:
                    zero = _value_to_literal(0, eqn.outvars[0].aval.dtype)
                    invs = list(new_invars) + [zero]
                    new_eqn = new_jaxpr_eqn(invs, list(eqn.outvars), add_p, {})
                else:
                    if eqn.params['type'] == add_p:
                        inv = list(new_invars)[0]
                        outv = gensym_fn(inv.aval)
                        div_eqn = new_jaxpr_eqn([
                            inv,
                            _value_to_literal(num_devices, inv.aval.dtype)
                        ], [outv], div_p, {})
                        new_eqns.append(div_eqn)
                        new_invars = [outv]
                    new_eqn = new_jaxpr_eqn(list(new_invars), list(eqn.outvars),
                                            eqn.primitive, dict(eqn.params))
                new_eqns.append(new_eqn)
            else:
                new_eqns.append(eqn)
            for v in eqn.outvars:
                if not isinstance(v, DropVar):
                    vars.add(v)
        return clone_jaxpr(closed_jaxpr, eqns=new_eqns)


def _no_allreduce(eqns):
    for eqn in eqns:
        if eqn.primitive == cross_mesh_allreduce_p:
            return False
    return True


def slice_apply_gradient(closed_jaxpr: ClosedJaxpr, grad_mesh: Dict[Var, int],
                         outvar_mesh: Dict[Var, OrderedSet[int]], num_mesh,
                         num_stage, donation_mapping: Dict[Var, Var], gensym_fn,
                         skip_cross_mesh_allreduce, mesh_num_devices):
    """
    Slice the apply gradient jaxpr based on mesh allocation information.

    Args:
        closed_jaxpr: closed jaxpr of apply_gradient function.
        grad_mesh: some invars should be at certain mesh;
            If not in the dict, the variable should be a global parameter.
        outvar_mesh: some outvars should be at certain mesh.
        num_mesh: number of meshes. If a mesh does not have apply gradient
          computation, add an empty jaxpr
        num_stage: number of stages in the apply gradient computation.
        donation_mapping: donation mapping for global invars
        skip_cross_mesh_allreduce: Skip cross mesh allreduce in profiling.

    Returns:
        jaxprs(List[ClosedJaxpr]): The i-th ClosedJaxpr runs at the i-th
          cluster.
        info: A tuple of:
            deps (List[Tuple[int, int]]): dependencies of apply gradient
              computations
            infered_global_invars (Dict[Var, List[int]]): From invar index to
              meshes need this invar.
    """
    var_mesh = {var: OrderedSet([mesh]) for var, mesh in grad_mesh.items()}
    for var in outvar_mesh:
        var_mesh.setdefault(var, OrderedSet()).update(outvar_mesh[var])
    # TODO(yonghao): running the split multiple times until no new splits
    closed_jaxpr, groups = ApplyGradRewriter(closed_jaxpr,
                                             var_mesh).split_replicated_eqns(
                                                 gensym_fn, num_mesh)
    # propagate to get var_at_mesh
    eqn_mesh, var_mesh = _propagate_var_at_mesh(closed_jaxpr.eqns, var_mesh)
    changed = True
    while changed:
        changed = _reverse_propagate_var_at_mesh(closed_jaxpr, donation_mapping,
                                                 eqn_mesh, var_mesh)

    sliced_eqns = [[] for _ in range(num_mesh)]
    for eqn_idx, eqn in enumerate(closed_jaxpr.eqns):
        if eqn_mesh[eqn_idx]:
            assert len(eqn_mesh[eqn_idx])
            for mesh in eqn_mesh[eqn_idx]:
                sliced_eqns[mesh].append(eqn)
        else:
            # all inputs are infered, all outputs are not assigned
            # TODO(yonghao): round-robin instead of using fixed index 0
            sliced_eqns[0].append(eqn)
            logger.debug(f'{eqn} are arbitrarily assigned')
            for invar in eqn.invars:
                if isinstance(invar, Var):
                    var_mesh.setdefault(invar, OrderedSet()).add(0)
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar):
                    assert (not var_mesh.setdefault(outvar, OrderedSet()) or
                            (len(var_mesh[outvar]) == 1 and
                             var_mesh[outvar][0] == 0))
                    var_mesh[outvar].add(0)

    # grouping invars and outvars
    (var_info,
     infered_global_invars) = _apply_grad_group_vars(closed_jaxpr, var_mesh,
                                                     num_mesh)
    invars, outvars, consts, constvars = var_info

    jaxprs = []
    mesh_assignment = {}

    for i in range(num_mesh):
        if not outvars[i] and _no_allreduce(sliced_eqns[i]):
            continue
        computation_idx = num_stage + len(jaxprs)
        # assign the current computation into mesh i
        mesh_assignment[computation_idx] = i
        sliced = Jaxpr(constvars[i], invars[i], outvars[i], sliced_eqns[i])
        closed_jaxpr = ClosedJaxpr(sliced, consts[i])
        num_devices = None if skip_cross_mesh_allreduce else mesh_num_devices[i]
        closed_jaxpr = ApplyGradRewriter.rewrite_allreduce(
            closed_jaxpr, skip_cross_mesh_allreduce, num_devices, gensym_fn)
        jaxprs.append(closed_jaxpr)

    info = mesh_assignment, infered_global_invars, groups
    return jaxprs, info


def apply_grad_add_marker(jaxprs: Sequence[ClosedJaxpr],
                          apply_in_to_acc_out: Dict[Var, Var],
                          gensym_fn,
                          computation=False):
    """Add pipeline markers for sliced apply grads, keep invars and outvars
    still unless.

    The invar is in apply_in_to_acc_out or invar is outvar:
    In the first case, the final invar follows the apply_in_to_acc_out;
    In the second case, the final outvar is recorded in outvar_map.

    Args:
        jaxprs: sliced apply grads.
        apply_in_to_acc_out: which output of accumulate grad corresponds to the
            invar of apply grad
        gensym_fn: gensym function of the whole jaxpr.
        computation: output JaxPipelineComputation or ClosedJaxpr.
    """
    results = []
    outvar_map = {}
    for i, jaxpr in enumerate(jaxprs):
        new_map = {}
        for invar in jaxpr.jaxpr.invars:
            if invar not in apply_in_to_acc_out:
                new_map[invar] = gensym_fn(invar.aval)
        for outvar in jaxpr.jaxpr.outvars:
            if not isinstance(outvar, Var):
                raise NotImplementedError(
                    'outvar of apply grad cannot be literal')
            if outvar in jaxpr.jaxpr.invars:
                if outvar not in outvar_map:
                    outvar_map[outvar] = gensym_fn(outvar.aval)
                continue
            new_map[outvar] = gensym_fn(outvar.aval)
        replaced = replace_all_with(jaxpr, new_map).jaxpr
        new_invars = list(
            map(lambda x: get_var_mapping(apply_in_to_acc_out, x),
                jaxpr.jaxpr.invars))
        new_outvars = list(
            map(lambda x: get_var_mapping(outvar_map, x), jaxpr.jaxpr.outvars))
        name = f'{i}_{APPLY_GRAD_MARKER_SUFFIX}'
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
            new_jaxpr = clone_jaxpr(jaxpr, new_invars, new_outvars, new_eqns)
            results.append(new_jaxpr)
    outvar_map.update(apply_in_to_acc_out)
    return results, outvar_map


def get_var_to_mesh(invars: Sequence[Var],
                    computations: Sequence[JaxPipelineComputation],
                    computation_to_mesh: Dict[int, int], apply_in_to_acc_out):
    """Get the mapping from variables to mesh."""
    # TODO(yonghao): now assume all gradients are variables(not literal)
    outvar2mesh = {}
    for i, computation in enumerate(computations):
        for var in computation.outvars:
            if isinstance(var, Var):
                outvar2mesh[var] = computation_to_mesh[i]
    return {
        invar: outvar2mesh[apply_in_to_acc_out[invar]]
        for invar in invars
        if ((invar in apply_in_to_acc_out) and
            (apply_in_to_acc_out[invar] in outvar2mesh))
    }

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
from alpa.util import (OrderedSet, clone_jaxpr, clone_jaxpr_eqn,
                       get_var_mapping, mesh_ids_hash, new_jaxpr_eqn,
                       slices_to_jaxpr)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# pylint: disable=redefined-builtin
unsafe_map, map = map, safe_map  # type: ignore
APPLY_GRAD_MARKER_SUFFIX = 'apply_grad'


def _filter_literal(vars):
    return [v for v in vars if isinstance(v, Var)]


def _filter_droped(vars):
    return [v for v in vars if not isinstance(v, DropVar)]


def _pipeline_marker_analysis(compute_eqns):
    """Get vars as inputs and outputs of layers"""
    layer_invars = set()
    pipeline_outvars = {}
    marker_cnt = 0
    for eqn in compute_eqns:
        if eqn.primitive is pipeline_p:
            if eqn.params['mark_type'] == 'end':
                for v in _filter_droped(eqn.outvars):
                    pipeline_outvars[v] = marker_cnt
                marker_cnt += 1
            elif eqn.params['mark_type'] == 'start':
                layer_invars.update(_filter_literal(eqn.invars))
    return layer_invars, pipeline_outvars


def _insert_to_pipeline_marker(marker, new_inv, mapping):
    invs = list(marker.invars)
    outvs = list(marker.outvars)
    for inv in new_inv:
        invs.append(inv)
        outvs.append(mapping[inv])
    return clone_jaxpr_eqn(marker, invs, outvs)


def _rewrite_compute_eqns(eqns, eqn_moved_to, gensym_fn):
    """Insert unmarked eqns(eqn_moved_to) to compute eqn sequence."""
    marker_cnt = 0
    new_eqns = []
    for eqn in eqns:
        if eqn.primitive is not pipeline_p:
            pass
        elif eqn.params['mark_type'] == 'start':
            cur_pipeline_start_idx = len(new_eqns)
        elif marker_cnt not in eqn_moved_to:
            marker_cnt += 1
        else:
            appended_eqns = eqn_moved_to[marker_cnt]
            i_marker = new_eqns[cur_pipeline_start_idx]
            o_marker = eqn
            layer_invar_map = {
                inv: outv
                for inv, outv in zip(i_marker.invars, i_marker.outvars)
                if isinstance(inv, Var) and not isinstance(outv, DropVar)
            }
            layer_outvar_map = {
                outv: inv
                for inv, outv in zip(o_marker.invars, o_marker.outvars)
                if isinstance(inv, Var) and not isinstance(outv, DropVar)
            }
            # collect and create all vars, then rewrite and create eqns
            inserted_invars = OrderedSet()
            inserted_outvars = OrderedSet()
            for eq in appended_eqns:
                # collect and create all used and output vars
                eq_new_invs = []
                for inv in eq.invars:
                    if isinstance(inv, Var):
                        if inv in layer_outvar_map:
                            # this layer defines the invar, use pre-marker ver.
                            eq_new_invs.append(layer_outvar_map[inv])
                        else:
                            if inv not in layer_invar_map:
                                # add new invar from other layers
                                layer_invar_map[inv] = gensym_fn(inv.aval)
                                inserted_invars.add(inv)
                            eq_new_invs.append(layer_invar_map[inv])
                    else:
                        eq_new_invs.append(inv)
                eq_new_outvs = []
                for outv in eq.outvars:
                    if isinstance(outv, DropVar):
                        eq_new_outvs.append(outv)
                    else:
                        new_mapped = gensym_fn(outv.aval)
                        layer_outvar_map[outv] = new_mapped
                        inserted_outvars.add(new_mapped)
                        eq_new_outvs.append(new_mapped)
                # create the new eqn
                new_eqns.append(clone_jaxpr_eqn(eq, eq_new_invs, eq_new_outvs))

            # create the new in marker
            new_eqns[cur_pipeline_start_idx] = _insert_to_pipeline_marker(
                i_marker, inserted_invars, layer_invar_map)
            layer_outvar_map = {v: k for k, v in layer_outvar_map.items()}
            eqn = _insert_to_pipeline_marker(o_marker, inserted_outvars,
                                             layer_outvar_map)
            marker_cnt += 1

        new_eqns.append(eqn)
    return new_eqns


def _get_delayed_eqns(compute_eqns, layer_invars, pipeline_outvars, gensym_fn):
    """
    Get eqns that can be delayed to apply gradient stage and rewrite eqns that
    cannot do so by moving them into a layer.

    An example of cannot delayed vars is: x is computed in layer0, and sent to
    layer1 and layer2. There is grad(x) = grad_1(x) + grad_2(x), but the
    grad(weight) depends on grad(x) and is in the acc_grad period, so we cannot
    delay it to the apply_grad period.
    """
    cross_layer_grad_eqns = []
    new_compute_eqns = []
    moved_to_layer_eqns = []

    marked_vars = set()
    used_vars = set()
    out_marker = True
    for eqn in reversed(compute_eqns):
        invars = _filter_literal(eqn.invars)
        outvars = _filter_droped(eqn.outvars)
        used_outvars = used_vars.intersection(outvars)
        if eqn.primitive is pipeline_p:
            # invars of a pipeline end marker is marked
            if eqn.params['mark_type'] == 'end':
                marked_vars.update(invars)
                out_marker = False
            else:
                out_marker = True
            new_compute_eqns.append(eqn)
        else:
            # we don't want to do dce here, because it may make its operand be
            # considered as cross layer grad, and then moved across microbatch
            # boundary, which is harder to analyze.
            if len(outvars) == 0 and out_marker:
                continue
            # only if an eqn is not used and is out marker will be it moved
            # after microbatch boundary. Those inside a microbatch boundary is
            # handled by later DCE.
            elif not used_outvars and out_marker:
                cross_layer_grad_eqns.append(eqn)
                continue
            elif marked_vars.issuperset(used_outvars):
                # eqn is marked if all outvars are marked, then mark its invars.
                marked_vars.update(invars)
                new_compute_eqns.append(eqn)
            else:
                assert not marked_vars.intersection(
                    outvars), f"'{eqn}' is partially marked."
                if layer_invars.intersection(outvars):
                    # move the marked var to the latest stage producing some of
                    # its invars.
                    moved_to_layer_eqns.append(eqn)
                    # update layer invars and marked vars.
                    layer_invars.update(invars)
                    marked_vars.update(outvars)
                else:
                    cross_layer_grad_eqns.append(eqn)
                    continue
        used_vars.update(invars)

    new_compute_eqns = list(reversed(new_compute_eqns))
    cross_layer_grad_eqns = list(reversed(cross_layer_grad_eqns))
    eqn_moved_to = {}
    for eqn in reversed(moved_to_layer_eqns):
        invars = _filter_literal(eqn.invars)
        outvars = _filter_droped(eqn.outvars)
        moved_to = max(pipeline_outvars[v] for v in invars)
        eqn_moved_to.setdefault(moved_to, []).append(eqn)
        pipeline_outvars.update({v: moved_to for v in outvars})
    if eqn_moved_to:
        new_compute_eqns = _rewrite_compute_eqns(new_compute_eqns, eqn_moved_to,
                                                 gensym_fn)
    return cross_layer_grad_eqns, new_compute_eqns


def _rewrite_microbatch_bound(microbatch_bound, delayed_eqns, gensym_fn):
    """
    Rewrite the microbatch bound because some eqns are moved from microbatched
    part of the graph to non-microbatched part.
    """
    microbatch_bound_in_to_outs = {}
    for invar, outvar in zip(microbatch_bound.invars, microbatch_bound.outvars):
        if isinstance(invar, Var) and not isinstance(outvar, DropVar):
            microbatch_bound_in_to_outs[invar] = outvar
    delayed_invars = OrderedSet()
    delayed_outvars = OrderedSet()
    for eqn in delayed_eqns:
        delayed_invars.update(_filter_literal(eqn.invars))
        delayed_outvars.update(_filter_droped(eqn.outvars))
    delayed_invars.difference_update(delayed_outvars)
    delayed_invars.difference_update(microbatch_bound_in_to_outs.keys())
    delayed_outvars.intersection_update(microbatch_bound_in_to_outs.keys())
    for invar in delayed_invars:
        microbatch_bound_in_to_outs[invar] = gensym_fn(invar.aval)
    # rewrite the microbatch_bound
    new_microbatch_bound_invars = []
    new_microbatch_bound_outvars = []
    for idx, var in enumerate(microbatch_bound.invars + list(delayed_invars)):
        # remove vars now defined after microbatch_bound.
        if isinstance(var, Var) and var in delayed_outvars:
            continue
        new_microbatch_bound_invars.append(var)
        # add vars now used after microbatch_bound.
        new_microbatch_bound_outvars.append(
            microbatch_bound.outvars[idx] if idx < len(microbatch_bound.invars)
            else microbatch_bound_in_to_outs[var])
    new_microbatch_bound = clone_jaxpr_eqn(microbatch_bound,
                                           new_microbatch_bound_invars,
                                           new_microbatch_bound_outvars)
    return new_microbatch_bound, microbatch_bound_in_to_outs


def _rewrite_delayed_gradient_sum_eqns(delayed_eqns,
                                       microbatch_bound_in_to_outs):
    """Change args of eqns that are delayed to the non-microbatched part."""
    new_apply_eqns = []
    for eqn in delayed_eqns:
        invars = [
            microbatch_bound_in_to_outs[var] if isinstance(var, Var) and
            var in microbatch_bound_in_to_outs else var for var in eqn.invars
        ]
        outvars = [
            microbatch_bound_in_to_outs[var] if not isinstance(var, DropVar) and
            var in microbatch_bound_in_to_outs else var for var in eqn.outvars
        ]
        new_apply_eqns.append(clone_jaxpr_eqn(eqn, invars, outvars))
    return new_apply_eqns


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
    layer_invars, pipeline_outvars = _pipeline_marker_analysis(compute_eqns)
    # Those eqn directly use output of pipeline end is delayed to apply grad.
    cross_layer_grad_eqns, new_compute_eqns = _get_delayed_eqns(
        compute_eqns, layer_invars, pipeline_outvars, gensym_fn)
    # Rewrite microbatch_bound and cross_layer_grad eqns.
    (new_microbatch_bound,
     microbatch_bound_in_to_outs) = _rewrite_microbatch_bound(
         microbatch_bound, cross_layer_grad_eqns, gensym_fn)
    # rewrite cross layer grad eqns and insert them to the top of apply eqns.
    new_apply_eqns = _rewrite_delayed_gradient_sum_eqns(
        cross_layer_grad_eqns, microbatch_bound_in_to_outs)
    new_apply_eqns += apply_eqns
    new_global_outvars = list(closed_jaxpr.jaxpr.outvars)
    for idx in range(len(new_global_outvars)):
        var = new_global_outvars[idx]
        if isinstance(var, Literal):
            continue
        if isinstance(var, Var) and var in microbatch_bound_in_to_outs:
            new_global_outvars[idx] = microbatch_bound_in_to_outs[var]
    closed_jaxpr = clone_jaxpr(closed_jaxpr,
                               eqns=new_compute_eqns + [new_microbatch_bound] +
                               new_apply_eqns,
                               outvars=new_global_outvars)
    return closed_jaxpr


def _remove_replicated_marked_var(closed_jaxpr: ClosedJaxpr):
    """Some variables are marked multiple times with the same marker.
    This pass removes them.
    """
    new_eqns = []
    var_map = {}
    mb_idx = None
    for eqn in closed_jaxpr.eqns:
        if eqn.primitive == pipeline_p:
            eqn_map = {}
            new_invars = []
            new_outvars = []
            if eqn.params['mark_type'] == 'grad':
                mb_idx = len(new_eqns)
            for inv, outv in zip(eqn.invars, eqn.outvars):
                if isinstance(outv, DropVar):
                    continue
                if isinstance(inv, Var):
                    if inv in var_map:
                        var_map[outv] = var_map[inv]
                        continue
                    elif inv in eqn_map:
                        var_map[outv] = eqn_map[inv]
                        continue
                if isinstance(inv, Var):
                    eqn_map[inv] = outv
                new_invars.append(inv)
                new_outvars.append(outv)
            new_eqns.append(clone_jaxpr_eqn(eqn, new_invars, new_outvars))
            continue
        new_invars = [get_var_mapping(var_map, v) for v in eqn.invars]
        new_eqns.append(clone_jaxpr_eqn(eqn, new_invars))
    sliced_eqns = new_eqns[:mb_idx], [new_eqns[mb_idx]], new_eqns[mb_idx + 1:]
    new_outvars = [
        get_var_mapping(var_map, v) for v in closed_jaxpr.jaxpr.outvars
    ]
    return clone_jaxpr(closed_jaxpr, outvars=new_outvars,
                       eqns=new_eqns), sliced_eqns


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
    # Locate the marker
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
    # Some equations are not marked. This pass moves them either into apply grad
    # or a layer.
    closed_jaxpr = _rewrite_cross_layer_grad(*sliced_eqns, gensym_fn,
                                             closed_jaxpr)
    closed_jaxpr, sliced_eqns = _remove_replicated_marked_var(closed_jaxpr)
    # Reconstruct jaxpr
    sliced_jaxprs = slices_to_jaxpr(closed_jaxpr, sliced_eqns)
    compute_grad, _, apply_grad = sliced_jaxprs  # pylint: disable=unbalanced-tuple-unpacking
    split_eqn = sliced_eqns[1][0]
    if len(apply_grad.eqns) == 0:
        logger.warning(
            'the apply gradient part is empty. Hint: apply() after alpa.grad')
    assert len(split_eqn.invars) == len(split_eqn.outvars)
    invars_without_dropvar = []
    outvars_without_dropvar = []
    for invar, outvar in zip(split_eqn.invars, split_eqn.outvars):
        if not isinstance(outvar, DropVar):
            invars_without_dropvar.append(invar)
            outvars_without_dropvar.append(outvar)
    split_eqn = clone_jaxpr_eqn(split_eqn, invars_without_dropvar,
                                outvars_without_dropvar)
    return closed_jaxpr, compute_grad, apply_grad, split_eqn


def _get_post_to_pre_marker_mapping(compute_jaxpr):
    """
    Get a dict that maps an out_var of a pipeline marker to
    its corresponding in_var.
    """
    post_marker_outs = _filter_droped(compute_jaxpr.jaxpr.outvars)
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
                           stage_to_mesh, gensym_func, num_meshes,
                           global_invars, global_outvars, donated_invars,
                           profiling, mesh_num_devices):
    """Slice apply_grad jaxpr into stages and assign them to the corresponding
    meshes."""
    # Process apply gradient:
    # change invars of apply grad to outvars of accumulate grad
    gradients = microbatch_bound.outvars
    apply_in_to_acc_out = dict(zip(gradients, microbatch_bound.invars))

    gradvar_to_mesh = get_var_to_mesh(gradients, pipeline_stages, stage_to_mesh,
                                      apply_in_to_acc_out)

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

    sliced_apply_grad_stages, apply_grad_placement, allreduce_groups = (
        slice_apply_gradient(apply_grad_jaxpr, gradvar_to_mesh, outvar_mesh,
                             num_meshes, len(pipeline_stages), donation_mapping,
                             gensym_func, profiling, mesh_num_devices))
    sliced_apply_grad_stages, out_map = apply_grad_add_marker(
        sliced_apply_grad_stages,
        apply_in_to_acc_out,
        gensym_func,
        computation=True)
    global_outvars = [get_var_mapping(out_map, var) for var in global_outvars]

    return (sliced_apply_grad_stages, apply_grad_placement, global_outvars,
            allreduce_groups)


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


def apply_grad_get_mean(apply_grad_jaxpr, global_outvars, gradients, gensym_fn,
                        num_microbatch, reduce_invars):
    """
    Get the mean of input (accumulated) gradients and run apply gradient.

    If the input is output, after this transform it outputs the divided version.
    """
    mapping = {}
    new_eqns = []
    invar_set = OrderedSet(apply_grad_jaxpr.jaxpr.invars)
    outvar_set = OrderedSet(apply_grad_jaxpr.jaxpr.outvars)
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
    replaced = replace_all_with(apply_grad_jaxpr, mapping)
    final_invars = list(apply_grad_jaxpr.jaxpr.invars)
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
    new_jaxpr = clone_jaxpr(apply_grad_jaxpr, final_invars, final_outvars,
                            new_eqns)
    global_outvars = [get_var_mapping(mapping, var) for var in global_outvars]
    return new_jaxpr, global_outvars


cross_mesh_allreduce_p = Primitive('__builtin$CrossMeshAllReduce')
_primitive_to_str = {add_p: b'SUM', and_p: b'AND', or_p: b'OR'}


def _cross_mesh_allreduce_xla_translation(c, *args, **kwargs):
    call_name = b'__builtin$CrossMeshAllReduce'
    assert len(args) == 1
    input_params = args[0]
    input_shape = c.get_shape(input_params)
    op_type = _primitive_to_str[kwargs['type']]
    opaque = op_type + b';' + mesh_ids_hash(kwargs['group_meshes'])

    # TODO(yonghao): the has_side_effect is to prevent CSE of the allreduce.
    # It might be replaced by adding its outvar to output
    sharding = xc.OpSharding()
    sharding.type = sharding.type.REPLICATED
    c.set_sharding(sharding)
    output = xc.ops.CustomCall(c,
                               call_name,
                               operands=(input_params,),
                               shape=input_shape,
                               has_side_effect=True,
                               opaque=opaque)
    c.clear_sharding()
    return output


xla.translations[cross_mesh_allreduce_p] = _cross_mesh_allreduce_xla_translation


def _init_eqn_var_mesh(closed_jaxpr, var_mesh):
    eqn_mesh = []
    var_mesh = dict(var_mesh)
    for eqn_idx, eqn in enumerate(closed_jaxpr.eqns):
        eqn_mesh.append(OrderedSet())
        for var in eqn.invars:
            if isinstance(var, Var):
                var_mesh.setdefault(var, OrderedSet())
        for var in eqn.outvars:
            if not isinstance(var, DropVar):
                var_mesh.setdefault(var, OrderedSet())
        if eqn.primitive != cross_mesh_allreduce_p:
            continue
        mesh_ids = eqn.params['group_meshes']
        for var, mesh_id in zip(eqn.invars, mesh_ids):
            var_mesh[var].add(mesh_id)
        var_mesh[eqn.outvars[0]] = OrderedSet(mesh_ids)
        eqn_mesh[eqn_idx] = OrderedSet(mesh_ids)
    return eqn_mesh, var_mesh


def _propagate_with_donation(closed_jaxpr, donation_mapping, var_mesh):
    changed = False
    for invar in closed_jaxpr.jaxpr.invars:
        if invar in donation_mapping:
            outvar = donation_mapping[invar]
            outvar_at = var_mesh[outvar]
            invar_at = var_mesh[invar]
            if invar_at.difference(outvar_at):
                outvar_at.update(invar_at)
                changed = True
            if outvar_at.difference(invar_at):
                invar_at.update(outvar_at)
    return changed


def _reverse_propagate_var_at_mesh(closed_jaxpr, donation_mapping, eqn_mesh,
                                   var_mesh):
    """Propagate var_at_mesh from output to make sure all operands are ready."""
    # Different from forward propagation, the eqn should be at to any mesh of
    # any outvar. Now the semantic switches from 'can be at' to 'is at'
    changed = False
    for reversed_idx, eqn in enumerate(reversed(closed_jaxpr.eqns)):
        eqn_idx = len(closed_jaxpr.eqns) - 1 - reversed_idx
        post_at_mesh = eqn_mesh[eqn_idx]
        at_mesh = OrderedSet()
        for outvar in eqn.outvars:
            if not isinstance(outvar, DropVar):
                at_mesh.update(var_mesh[outvar])
        if not at_mesh:
            continue
        if (not post_at_mesh or at_mesh.difference(post_at_mesh)):
            changed = True
            post_at_mesh.update(at_mesh)
            if eqn.primitive != cross_mesh_allreduce_p:
                for invar in eqn.invars:
                    if isinstance(invar, Var):
                        var_mesh[invar].update(at_mesh)
    changed |= _propagate_with_donation(closed_jaxpr, donation_mapping,
                                        var_mesh)
    return changed


def _forward_propagate_at_mesh(closed_jaxpr, eqn_mesh, var_mesh, aggressive):
    """
    Propagate the can/may be at info for eqns and vars not yet allocated.

    Can at mode is conservative. It computes the intersection of all invars'
    meshes. When var_0 is at mesh_0 and var_1 at mesh_0,1, the eqn can only be
    at mesh 0.

    May at mode is to handle those cannot be solved by can at mode. That is,
    at one point, the intersection of all invars' meshes is empty. Then there
    should have some redundant computation and memory consumptions.

    TODO: Currently we only use the first element of all available candidates in
    both mode, but for 'may at' mode, we need to pick the one with the least
    redundancy using some estimation. For 'can at' mode, a round-robin is better
    """
    var_infered_at = {}
    for eqn_idx, eqn in enumerate(closed_jaxpr.eqns):
        if eqn_mesh[eqn_idx]:
            continue
        eqn_infered_at = None
        # For invar_0 available at mesh_0, invar_1 available at mesh_0,1
        # the outvar is better at mesh_0 instead of mesh_0,1
        for var in eqn.invars:
            if not isinstance(var, Var):
                continue
            if var_mesh[var]:
                invar_infered_at = var_mesh[var]
            elif var in var_infered_at and var_infered_at[var]:
                invar_infered_at = var_infered_at[var]
            else:
                invar_infered_at = None
            if invar_infered_at:
                if eqn_infered_at is None:
                    eqn_infered_at = OrderedSet(invar_infered_at)
                else:
                    if aggressive:
                        eqn_infered_at.update(invar_infered_at)
                    else:
                        eqn_infered_at.intersection_update(invar_infered_at)
        if eqn_infered_at:
            for var in eqn.outvars:
                if not isinstance(var, DropVar):
                    var_infered_at[var] = OrderedSet(eqn_infered_at)
    changed = False
    for var in closed_jaxpr.jaxpr.outvars:
        if (not isinstance(var, DropVar) and not var_mesh[var]):
            if var in var_infered_at:
                var_mesh[var] = OrderedSet([list(var_infered_at[var])[0]])
            elif aggressive:
                var_mesh[var] = OrderedSet([0])
            else:
                continue
            changed = True
    return changed


def _apply_grad_group_vars(closed_jaxpr: ClosedJaxpr, var_mesh, num_mesh):
    """Slice the input, output and consts of the jaxpr based on var_mesh."""
    global_invars = closed_jaxpr.jaxpr.invars
    invars = [[] for _ in range(num_mesh)]
    outvars = [[] for _ in range(num_mesh)]
    constvars = [[] for _ in range(num_mesh)]
    consts = [[] for _ in range(num_mesh)]
    # grouping invars and outvars
    for invar in global_invars:
        for mesh in var_mesh[invar]:
            invars[mesh].append(invar)
    for outvar in closed_jaxpr.jaxpr.outvars:
        for mesh in var_mesh[outvar]:
            outvars[mesh].append(outvar)
    # grouping consts and constvars
    for aval, var in zip(closed_jaxpr.consts, closed_jaxpr.jaxpr.constvars):
        for mesh in var_mesh[var]:
            consts[mesh].append(aval)
            constvars[mesh].append(var)
    return invars, outvars, consts, constvars


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
        """An eqn is reducable if it is a reducable and scalar operation"""
        # the is_scalar is to avoid a large all-reduce for tied-embedding
        # it can be improved by adding computation-communication tradeoff
        return (eqn.primitive in _reducable_operators and
                eqn.outvars[0].aval.shape == ())

    def _forward_propagate(self):
        """
        A conservative propagation that stops when the eqn's invars are from
        multiple meshes.
        """
        self.eqn_mesh = {}
        self.var_use = {}
        self.var_def = {}
        for eqn_idx, eqn in enumerate(self.eqns):
            for invar in _filter_literal(eqn.invars):
                self.var_use.setdefault(invar, OrderedSet()).add(eqn_idx)
            for outvar in _filter_droped(eqn.outvars):
                self.var_def[outvar] = eqn_idx
        has_color = OrderedSet([
            self.var_def[k]
            for k in self.var_mesh
            if (len(self.var_mesh[k]) > 0 and k in self.var_def)
        ])
        q = list(has_color)
        while len(q) > 0:
            for outv in _filter_droped(self.eqns[q[0]].outvars):
                if outv not in self.var_use:
                    continue
                used_eqns = self.var_use[outv]
                has_color.update(used_eqns)
                for e_id in used_eqns.difference(has_color):
                    q.append(e_id)
            q = q[1:]

        # Propagate the first round
        for eqn_idx, eqn in enumerate(self.eqns):
            at_mesh = OrderedSet()
            for invar in _filter_literal(eqn.invars):
                at_mesh.update(self.var_mesh.setdefault(invar, OrderedSet()))
            # TODO(yonghao): round robin this and use it in later positions
            if len(at_mesh) == 0 and eqn_idx not in has_color:
                at_mesh = OrderedSet([0])
            if len(at_mesh) == 1:
                for invar in _filter_literal(eqn.invars):
                    self.var_mesh.setdefault(invar,
                                             OrderedSet()).update(at_mesh)
            self.eqn_mesh[eqn_idx] = list(at_mesh)
            for outvar in _filter_droped(eqn.outvars):
                self.var_mesh[outvar] = OrderedSet(at_mesh)

    def _reducable_chain_lookup(self, eqn_idx, num_mesh):
        """
        Pattern matching. For y = x_0 op x_1 op x_2 ... op x_n, it is as
        y_0 = x_0 op x_1, y_1 = y_0 op x_2, ... in jaxpr. This function collects
        all such x_0, x_1, ... x_n by making sure that intermediates like y_0 &
        y_1 are not used elsewhere.

        Returns:
            mesh_vars: list of variables being reduced in a certain mesh.
            final_var: The final outvar(the y above)
            removed: Indices of eqns being removed. They compute intermediates.
            literals: Literals along with the reduction
        """
        # List[mesh_idx -> List[Vars]]
        mesh_vars = [[] for _ in range(num_mesh)]
        literals = []
        eqn = self.eqns[eqn_idx]
        nxt_idx, nxt_eqn = eqn_idx, eqn
        reducable_chain = []
        while self._reducable(nxt_eqn) and (nxt_eqn.primitive == eqn.primitive):
            cur_idx, cur_eqn = nxt_idx, nxt_eqn
            reducable_chain.append(cur_idx)
            outv_use = self.var_use.setdefault(cur_eqn.outvars[0], OrderedSet())
            # If the var is used in multiple places or global output, it is not
            # a safe intermediate variable and the chain ends.
            if len(outv_use) != 1 or cur_eqn.outvars[0] in self.outvars:
                break
            nxt_idx = list(outv_use)[0]
            nxt_eqn = self.eqns[nxt_idx]
        if cur_idx == eqn_idx:
            return None, None, None, None
        final_var = cur_eqn.outvars[0]
        # split eqns on the reducable chain into meshes
        reducable_set = set(reducable_chain)
        for reduced_idx in reducable_chain:
            reduced_eqn = self.eqns[reduced_idx]
            for op in reduced_eqn.invars:
                # We can assign all literals to mesh 0 cuz they'll be optimized
                # by arithmetic simplification.
                if isinstance(op, Literal):
                    mesh_vars[0].append(op)
                    continue
                def_idx = self.var_def[op]
                if def_idx not in reducable_set:
                    def_meshes = self.eqn_mesh[def_idx]
                    # TODO(yonghao): round-robin this
                    mesh_vars[list(def_meshes)[0]].append(op)
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
                    # This is the first var in the mesh for the chain
                    cur_val = v
                    continue
                new_var = gensym_fn(cur_val.aval)
                # accumulate in-mesh result
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
            new_jaxpr_eqn(allreduce_vars, [outvar], cross_mesh_allreduce_p, {
                'type': primitive,
                'group_meshes': mesh_ids
            }))
        return appended_eqns, mesh_ids

    def split_replicated_eqns(self, gensym_fn, num_mesh):
        """Rewrite apply grad jaxpr to eqns so as to """
        self._forward_propagate()
        new_eqns_before_var = {}
        # Try to match the pattern
        removed_eqns = set()
        allreduce_groups = OrderedSet()
        for eqn_idx, eqn in enumerate(self.eqns):
            if eqn_idx in removed_eqns:
                continue
            if (eqn_idx in self.eqn_mesh and len(self.eqn_mesh[eqn_idx]) > 1 and
                    self._reducable(eqn)):
                (mesh_vars, final_var, removed,
                 literals) = self._reducable_chain_lookup(eqn_idx, num_mesh)
                if mesh_vars is None:
                    # Only one eqn matches the pattern, skip it
                    continue
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
        mesh_assignment(Dict[int, int]): The i-th ClosedJaxpr runs at the
          mesh_assignment[i]-th cluster.
        allreduce_groups(Tuple[Tuple[int]]): Groups of mesh ids that need to
          be in the same allreduce group to perform cross-mesh allreduce.
    """
    var_mesh = {var: OrderedSet([mesh]) for var, mesh in grad_mesh.items()}
    for var in outvar_mesh:
        var_mesh.setdefault(var, OrderedSet()).update(outvar_mesh[var])
    # TODO(yonghao): running the split multiple times until no new splits
    closed_jaxpr, allreduce_groups = ApplyGradRewriter(
        closed_jaxpr, var_mesh).split_replicated_eqns(gensym_fn, num_mesh)
    eqn_mesh, var_mesh = _init_eqn_var_mesh(closed_jaxpr, var_mesh)
    changed = True
    _propagate_with_donation(closed_jaxpr, donation_mapping, var_mesh)
    while changed:
        changed = _reverse_propagate_var_at_mesh(closed_jaxpr, donation_mapping,
                                                 eqn_mesh, var_mesh)
    changed = _forward_propagate_at_mesh(closed_jaxpr, eqn_mesh, var_mesh,
                                         False)
    while changed:
        changed = _reverse_propagate_var_at_mesh(closed_jaxpr, donation_mapping,
                                                 eqn_mesh, var_mesh)
    changed = _forward_propagate_at_mesh(closed_jaxpr, eqn_mesh, var_mesh, True)
    while changed:
        changed = _reverse_propagate_var_at_mesh(closed_jaxpr, donation_mapping,
                                                 eqn_mesh, var_mesh)

    sliced_eqns = [[] for _ in range(num_mesh)]
    for eqn_idx, eqn in enumerate(closed_jaxpr.eqns):
        if eqn_mesh[eqn_idx]:
            for mesh in eqn_mesh[eqn_idx]:
                sliced_eqns[mesh].append(eqn)

    # grouping invars and outvars
    invars, outvars, consts, constvars = _apply_grad_group_vars(
        closed_jaxpr, var_mesh, num_mesh)

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

    return jaxprs, mesh_assignment, allreduce_groups


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
        new_invars = [
            get_var_mapping(apply_in_to_acc_out, var)
            for var in jaxpr.jaxpr.invars
        ]
        new_outvars = [
            get_var_mapping(outvar_map, var) for var in jaxpr.jaxpr.outvars
        ]
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

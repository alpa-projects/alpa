"""Pipeline primitive definitions."""
import numpy as np

from jax import lax
from jax.core import Primitive, abstract_unit, new_jaxpr_eqn, dropvar
from jax.interpreters import xla, ad
from jax.lib import xla_client as xc
from jax.tree_util import tree_flatten, tree_unflatten

from parax.pipeline_parallel.xla_custom_call_marker import xla_pipeline_marker, identity

xc.register_custom_call_target(b'xla_pipeline_marker',
                               xla_pipeline_marker(),
                               platform='gpu')
xc.register_custom_call_target(b'identity', identity(), platform='gpu')


def flatten_shape_byte_sizes(shape):

    def _flatten_shape_byte_sizes(shape):
        if shape.is_tuple():
            res = []
            for sub_shape in shape.tuple_shapes():
                res += _flatten_shape_byte_sizes(sub_shape)
            return res
        else:
            return [shape.numpy_dtype().itemsize * np.prod(shape.dimensions())]

    res = _flatten_shape_byte_sizes(shape)
    return np.array(res, dtype=np.int64)


def mark_pipeline_xla(c, *args):
    input_params = xc.ops.Tuple(c, args)
    input_shape = c.get_shape(input_params)
    flattened_byte_sizes = flatten_shape_byte_sizes(input_shape)
    output_tuple = xc.ops.CustomCallWithLayout(c,
                                               b'xla_pipeline_marker',
                                               operands=(input_params,),
                                               shape_with_layout=input_shape,
                                               operand_shapes_with_layout=(input_shape,),
                                               opaque=flattened_byte_sizes.tobytes())
    return output_tuple


# Define a Jax primitive to mark start/end of a pipeline stage.
pipeline_p = Primitive('pipeline')
pipeline_p.multiple_results = True


def mark_pipeline(*args, name: str, mark_type: str):
    """
    Mark the start/end of a pipeline stage.

    Args:
        *args: represents the pipeline input/output of a pipeline stage.
        name (str): Name of the pipeline stage.
        mark_type (str): start or end of a pipeline stage, can be "start",
            "end", "jvp_start", or "jvp_end". The latter two are used for
            backward pass.
    """
    if mark_type not in ('start', 'end', 'jvp_start', 'jvp_end'):
        raise ValueError('Unknown mark type: %s' % mark_type)
    return pipeline_p.bind(*args, name=name, mark_type=mark_type)


def mark_pipeline_jaxpreqn(invars, outvars, name: str, mark_type: str):
    if mark_type not in ('start', 'end', 'jvp_start', 'jvp_end'):
        raise ValueError('Unknown mark type: %s' % mark_type)
    if len(outvars) == 0:
        outvars = [dropvar]
    return new_jaxpr_eqn(invars, outvars, pipeline_p, {
        'name': name,
        'mark_type': mark_type
    })


def _pipeline_impl(*args, **kwargs):
    # The pipeline marker acts as an identity function.
    return args if len(args) > 0 else (None,)


def _pipeline_abstract_eval(*args, **kwargs):
    return args if len(args) > 0 else (abstract_unit,)


def _pipeline_xla_translation(c, *args, **kwargs):
    return mark_pipeline_xla(c, *args)


def _pipeline_value_and_jvp(arg_values, arg_tangents, name, mark_type):
    primal_outs = mark_pipeline(*arg_values, name=name, mark_type=mark_type)
    # TODO(zhuohan): Check the semantics here works for higher order gradients.
    if mark_type in ("start", "jvp_start"):
        tangent_mark_type = "jvp_start"
    elif mark_type in ("end", "jvp_end"):
        tangent_mark_type = "jvp_end"
    else:
        raise ValueError("Invalid mark_type")

    marker_inputs = []
    tan_marker_id = []
    for val, tan in zip(arg_values, arg_tangents):
        if isinstance(tan, ad.Zero):
            tan_marker_id.append(-1)
        else:
            tan_marker_id.append(len(marker_inputs))
            marker_inputs.append(tan)
    res = mark_pipeline(*marker_inputs,
                        name=name,
                        mark_type=tangent_mark_type)
    tangent_outs = []
    for i, (val, tan) in enumerate(zip(arg_values, arg_tangents)):
        if tan_marker_id[i] == -1:
            tangent_outs.append(ad.Zero(val.aval))
        else:
            tangent_outs.append(res[tan_marker_id[i]])

    return primal_outs, tangent_outs


def _pipeline_transpose(ct, *args, name, mark_type):
    # TODO(zhuohan): Check the semantics here works for higher order gradients.
    if mark_type in ("start", "jvp_start"):
        transposed_mark_type = "end"
    elif mark_type in ("end", "jvp_end"):
        transposed_mark_type = "start"
    else:
        raise ValueError("Invalid mark_type")
    marker_inputs = []
    ctan_marker_id = []
    for val, ctan in zip(args, ct):
        if isinstance(ctan, ad.Zero):
            ctan_marker_id.append(-1)
        else:
            ctan_marker_id.append(len(marker_inputs))
            marker_inputs.append(ctan)
    res = mark_pipeline(*marker_inputs, name=name, mark_type=transposed_mark_type)
    new_ct = []
    for i, (val, ctan) in enumerate(zip(args, ct)):
        if ctan_marker_id[i] == -1:
            new_ct.append(ad.Zero(val.aval))
        else:
            new_ct.append(res[ctan_marker_id[i]])
    return new_ct


pipeline_p.def_impl(_pipeline_impl)
pipeline_p.def_abstract_eval(_pipeline_abstract_eval)
xla.translations[pipeline_p] = _pipeline_xla_translation
ad.primitive_jvps[pipeline_p] = _pipeline_value_and_jvp
ad.primitive_transposes[pipeline_p] = _pipeline_transpose


def mark_gradient(grad):
    grad_flat, tree = tree_flatten(grad)
    grad_flat = pipeline_p.bind(*grad_flat, name='grad', mark_type='grad')
    grad = tree_unflatten(tree, grad_flat)
    return grad

"""Pipeline primitive definitions."""
import numpy as np

from jax.core import Primitive, abstract_unit
from jax.interpreters import xla, ad
from jax.lib import xla_client as xc

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


def _pipeline_impl(*args, **kwargs):
    # The pipeline marker acts as an identity function.
    return args if len(args) > 0 else (None,)


def _pipeline_abstract_eval(*args, **kwargs):
    return args if len(args) > 0 else (abstract_unit,)


def _pipeline_xla_translation(c, *args, **kwargs):
    return xc.ops.Tuple(c, args) if len(args) > 0 else xc.ops.Tuple(c, (xc.ops.Constant(c, np.float32(0.0)),))


def _pipeline_value_and_jvp(arg_values, arg_tangents, name, mark_type):
    primal_outs = mark_pipeline(*arg_values, name=name, mark_type=mark_type)
    # TODO(zhuohan): Check the semantics here works for higher order gradients.
    if mark_type in ("start", "jvp_start"):
        tangent_mark_type = "jvp_start"
    elif mark_type in ("end", "jvp_end"):
        tangent_mark_type = "jvp_end"
    else:
        raise ValueError("Invalid mark_type")
    tangent_outs = mark_pipeline(*arg_tangents, name=name, mark_type=tangent_mark_type)
    return primal_outs, tangent_outs


def _pipeline_transpose(ct, *args, name, mark_type):
    # TODO(zhuohan): Check the semantics here works for higher order gradients.
    if mark_type in ("start", "jvp_start"):
        transposed_mark_type = "end"
    elif mark_type in ("end", "jvp_end"):
        transposed_mark_type = "start"
    else:
        raise ValueError("Invalid mark_type")
    res = mark_pipeline(*ct, name=name, mark_type=transposed_mark_type)
    return res


pipeline_p.def_impl(_pipeline_impl)
pipeline_p.def_abstract_eval(_pipeline_abstract_eval)
xla.translations[pipeline_p] = _pipeline_xla_translation
ad.primitive_jvps[pipeline_p] = _pipeline_value_and_jvp
ad.primitive_transposes[pipeline_p] = _pipeline_transpose

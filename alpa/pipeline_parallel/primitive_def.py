"""Define a new Jax primitive pipeline_marker to mark the boundary of pipeline
computations."""
import numpy as np

from jax.core import Primitive
from jax.interpreters import xla, ad
from jax.lib import xla_client as xc
from jax.tree_util import tree_flatten, tree_unflatten

from alpa.util import new_jaxpr_eqn

########## Public APIs ##########

# Define a Jax primitive to mark start/end of a pipeline computation.
pipeline_p = Primitive("pipeline_marker")


def mark_pipeline_boundary():
    """Mark the boundary of pipeline layers. We reuse pipeline_marker for this
    functionality."""
    return pipeline_p.bind(name="boundary", mark_type="boundary")


def mark_gradient(grad):
    """Mark variables as gradients. We reuse pipeline_marker for this
    functionality."""
    grad_flat, tree = tree_flatten(grad)
    grad_flat = pipeline_p.bind(*grad_flat, name="grad", mark_type="grad")
    grad = tree_unflatten(tree, grad_flat)
    return grad


def mark_pipeline_jaxpreqn(invars, outvars, name: str, mark_type: str):
    """Make a new jaxpr equation."""
    if mark_type not in ("start", "end", "jvp_start", "jvp_end"):
        raise ValueError(f"Unknown mark type: {mark_type}")
    return new_jaxpr_eqn(invars, outvars, pipeline_p, {
        "name": name,
        "mark_type": mark_type
    })


def mark_hook_jaxpreqn(invars, outvars):
    """Mark some variables in a hook. We then extract the information
    of the variables in the hook."""
    return new_jaxpr_eqn(invars, outvars, pipeline_p, {
        "name": "hook",
        "mark_type": "hook"
    })


########## Internal Registration ##########
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


def xla_custom_call(c, call_name, op_name, *args):
    input_params = xc.ops.Tuple(c, args)
    input_shape = c.get_shape(input_params)
    flattened_byte_sizes = flatten_shape_byte_sizes(input_shape)
    op_metadata = xc.OpMetadata(op_name=op_name)
    c.set_op_metadata(op_metadata)

    if len(args) == 0:
        # If the custom call is an empty marker, it cannot be annotated
        # by sharding propagation, so we set a sharding for it.
        sharding = xc.OpSharding()
        sharding.type = sharding.type.REPLICATED
        c.set_sharding(sharding)

    if call_name == "pipeline_marker":
        output_tuple = xc.ops.CustomCall(
            c,
            b"pipeline_marker",
            operands=(input_params,),
            shape=input_shape,
            # Prevent the deletion of an empty marker
            has_side_effect=True,
            opaque=flattened_byte_sizes.tobytes())
    elif call_name == "optimization_barrier":
        output_tuple = xc.ops.OptimizationBarrier(input_params)
    else:
        raise ValueError("Invalid call_name: {call_name}")

    c.clear_op_metadata()
    c.clear_sharding()
    return output_tuple


def _pipeline_impl(*args, **kwargs):
    # pylint: disable=unused-argument
    # The pipeline marker acts as an identity function.
    return args


def _pipeline_abstract_eval(*args, **kwargs):
    # pylint: disable=unused-argument
    # The pipeline marker acts as an identity function.
    return args


def _pipeline_xla_translation(c, *args, **kwargs):
    name = kwargs["name"] + "$" + kwargs["mark_type"]
    if kwargs["name"] == "hook":
        call_name = "optimization_barrier"
    else:
        call_name = "pipeline_marker"

    return xla_custom_call(c, call_name, name, *args)


def _pipeline_value_and_jvp(arg_values, arg_tangents, name, mark_type):
    primal_outs = pipeline_p.bind(*arg_values, name=name, mark_type=mark_type)
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
    res = pipeline_p.bind(*marker_inputs,
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
    res = pipeline_p.bind(*marker_inputs,
                          name=name + "_backward",
                          mark_type=transposed_mark_type)
    new_ct = []
    for i, (val, ctan) in enumerate(zip(args, ct)):
        if ctan_marker_id[i] == -1:
            new_ct.append(ad.Zero(val.aval))
        else:
            new_ct.append(res[ctan_marker_id[i]])
    return new_ct


pipeline_p.def_impl(_pipeline_impl)
pipeline_p.def_abstract_eval(_pipeline_abstract_eval)
pipeline_p.multiple_results = True
xla.translations[pipeline_p] = _pipeline_xla_translation
ad.primitive_jvps[pipeline_p] = _pipeline_value_and_jvp
ad.primitive_transposes[pipeline_p] = _pipeline_transpose

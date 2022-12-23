"""User specified manual sharding strategy following pjit's api."""
import dataclasses
from typing import Any, Optional, OrderedDict, Tuple, Union

from jax._src.lib import xla_client as xc
from jax._src.tree_util import _replace_nones
from jax._src.util import safe_zip
from jax.experimental.pjit import (_is_unspecified, _is_auto, _is_from_gda,
                                   _prepare_axis_resources, get_array_mapping,
                                   _UNSPECIFIED)
from jax.interpreters import mlir, pxla, xla
from jax.tree_util import tree_unflatten, tree_flatten, tree_map

from alpa.util import undefined_sharding_spec_proto


@dataclasses.dataclass
class ManualShardingOption:
    """Options to manually set shardings in pjit convention."""
    mesh_axis_names: Tuple[pxla.MeshAxisName, ...] = None
    # According to pjit, None means replicated.
    in_axis_resources: Any = _UNSPECIFIED
    out_axis_resources: Any = _UNSPECIFIED


def _parsed_pspec_to_hlo_sharding(
    mesh_shape,
    mesh_axis_names,
    parsed_pspec,
    num_dimensions: int,
    axis_ctx: Optional[Union[mlir.SPMDAxisContext, mlir.ShardingContext]] = None
) -> xc.OpSharding:
    """
    TODO(yonghao): support auto(see how pxla.py lowers it)

    This function inlines _create_mesh_pspec_sharding_from_parsed_pspec and
    _process_in_axis_resources. It skips some checks there including
    _is_unspecified_or_from_gda_or_auto, pjit_check_aval_sharding. It also skips
    the local-global translation because we always assume alpa handles jaxprs at
    the driver side.
    """
    if _is_unspecified(parsed_pspec):
        return undefined_sharding_spec_proto()
    if _is_from_gda(parsed_pspec):
        raise NotImplementedError("alpa does not support global device array.")
    if _is_auto(parsed_pspec):
        raise NotImplementedError("")

    array_mapping = get_array_mapping(parsed_pspec)
    sharding_spec = pxla.new_mesh_sharding_specs(mesh_shape, mesh_axis_names)(
        num_dimensions, array_mapping)
    # Used in `with_sharding_constraint`.
    special_axes = {}
    # Manual axes is only used with xmap.
    # TODO: check whether this manual is conflict with what we use for the
    # unspecified type(pjit uses REPLICATED as unspecified)
    if axis_ctx is not None and isinstance(axis_ctx, mlir.SPMDAxisContext):
        axis_names = mesh_axis_names
        for manual_axis in axis_ctx.manual_axes:
            special_axes[axis_names.index(
                manual_axis)] = xc.OpSharding.Type.MANUAL
    op_sharding = sharding_spec.sharding_proto(special_axes=special_axes)
    return op_sharding


def flatten_axes(treedef, axis_tree):
    """Flatten the axis tree and consider None as an effective value."""
    proxy = object()
    dummy = tree_unflatten(treedef, [object()] * treedef.num_leaves)

    axes = []

    def add_leaves(i, x):
        axes.extend([i] * len(tree_flatten(x)[0]))

    tree_map(add_leaves, _replace_nones(proxy, axis_tree), dummy)
    axes = [None if a is proxy else a for a in axes]
    assert len(axes) == treedef.num_leaves
    return axes


def get_manual_sharding_spec(
        sharding_option: ManualShardingOption, mesh_shape, in_tree, out_tree,
        in_avals, out_avals) -> Tuple[Tuple[xc.OpSharding], xc.OpSharding]:
    """Create input and output sharding spec from user's in_axis_resources."""
    named_mesh_shape = OrderedDict(
        (name, size)
        for name, size in safe_zip(sharding_option.mesh_axis_names, mesh_shape))

    # process input
    if _is_unspecified(sharding_option.in_axis_resources):
        in_op_shardings = None
    else:
        in_axis_resources, _, _, any_auto = _prepare_axis_resources(
            sharding_option.in_axis_resources, "in_axis_resources")
        if any_auto:
            raise NotImplementedError(
                "auto mode in manual partition is unsupported.")
        in_axis_flat = tuple(flatten_axes(in_tree, in_axis_resources))
        if any(_is_unspecified(in_axis) for in_axis in in_axis_flat):
            assert all(_is_unspecified(in_axis) for in_axis in in_axis_flat)
        in_op_shardings = tuple(
            _parsed_pspec_to_hlo_sharding(named_mesh_shape,
                                          sharding_option.mesh_axis_names, axis,
                                          len(aval.shape))
            for axis, aval in safe_zip(in_axis_flat, in_avals))

    # process output
    if _is_unspecified(sharding_option.out_axis_resources):
        out_op_shardings = None
    else:
        out_axis_resources, _, _, _ = _prepare_axis_resources(
            sharding_option.out_axis_resources, "out_axis_resources")
        out_axis_flat = tuple(flatten_axes(out_tree, out_axis_resources))
        if any(_is_unspecified(out_axis) for out_axis in out_axis_flat):
            assert all(_is_unspecified(out_axis) for out_axis in out_axis_flat)
        out_op_shardings = tuple(
            _parsed_pspec_to_hlo_sharding(named_mesh_shape,
                                          sharding_option.mesh_axis_names, axis,
                                          len(aval.shape))
            for axis, aval in safe_zip(out_axis_flat, out_avals))
    return in_op_shardings, out_op_shardings

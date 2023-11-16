"""User specified manual sharding strategy following pjit's api."""
import dataclasses
from typing import Any, Optional, OrderedDict, Tuple, Union

from jax._src.lib import xla_client as xc
from jax.interpreters.pxla import _is_unspecified, _UNSPECIFIED
# _is_from_gda

from jax._src.sharding_impls import prepare_axis_resources, get_array_mapping, ParsedPartitionSpec
from jax._src.pjit import is_auto
from jax._src.tree_util import _replace_nones
from jax._src.util import safe_zip
from jax.interpreters import mlir, pxla
from jax.tree_util import tree_unflatten, tree_flatten, tree_map

from alpa.util import undefined_sharding_spec_proto


@dataclasses.dataclass
class ManualShardingOption:
    """Options to manually set shardings in pjit convention."""
    mesh_axis_names: Tuple[pxla.MeshAxisName, ...] = None
    submesh_axis_names: Tuple[Tuple[pxla.MeshAxisName, ...], ...] = None
    # According to pjit, None means replicated.
    in_axis_resources: Any = _UNSPECIFIED
    out_axis_resources: Any = _UNSPECIFIED


@dataclasses.dataclass
class ParsedManualShardingOption:
    """Options """
    mesh_axis_names: Tuple[pxla.MeshAxisName, ...] = None
    submesh_axis_names: Tuple[Tuple[pxla.MeshAxisName, ...], ...] = None
    # Parsed and flatten status
    in_parsed_pspec: Tuple[ParsedPartitionSpec, ...] = None
    out_parsed_pspec: Tuple[ParsedPartitionSpec, ...] = None


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
    # if _is_from_gda(parsed_pspec):
    #     raise NotImplementedError("alpa does not support global device array.")
    if is_auto(parsed_pspec):
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


def _flatten_axes(treedef, axis_tree):
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


def _prepare_axis_and_flatten(axis_resources, tree, name):
    parsed_axis_resources, _, _ = prepare_axis_resources(
        axis_resources, name)
    axis_flat = tuple(_flatten_axes(tree, parsed_axis_resources))
    if any(_is_unspecified(in_axis) for in_axis in axis_flat):
        assert all(_is_unspecified(in_axis) for in_axis in axis_flat)
    return axis_flat


def get_flatten_axis_resources(sharding_option: ManualShardingOption, in_tree,
                               out_tree) -> ParsedManualShardingOption:
    """Flatten axis resources for pipeline parallel to dispatch."""
    if sharding_option is None:
        return None

    # process input
    if _is_unspecified(sharding_option.in_axis_resources):
        in_axis_flat = None
    else:
        in_axis_flat = _prepare_axis_and_flatten(
            sharding_option.in_axis_resources, in_tree, "in_axis_resources")

    # process output
    if _is_unspecified(sharding_option.out_axis_resources):
        out_axis_flat = None
    else:
        out_axis_flat = _prepare_axis_and_flatten(
            sharding_option.out_axis_resources, out_tree, "out_axis_resources")
    return ParsedManualShardingOption(sharding_option.mesh_axis_names,
                                      sharding_option.submesh_axis_names,
                                      in_axis_flat, out_axis_flat)


def parsed_spec_to_opsharding(axes, avals, mesh_shape, mesh_axis_names):
    """Translate axis(a sequence of ParsedPartitionSpec) into OpShardings"""
    if axes is None:
        return None

    named_mesh_shape = OrderedDict(
        (name, size) for name, size in safe_zip(mesh_axis_names, mesh_shape))
    op_shardings = tuple(
        _parsed_pspec_to_hlo_sharding(named_mesh_shape, mesh_axis_names, axis,
                                      len(aval.shape))
        for axis, aval in safe_zip(axes, avals))
    return op_shardings


def get_manual_sharding_spec(
        sharding_option: ManualShardingOption, mesh_shape, in_tree, out_tree,
        in_avals, out_avals) -> Tuple[Tuple[xc.OpSharding, ...], xc.OpSharding]:
    """Create input and output sharding spec from user's in_axis_resources."""
    parsed_resources = get_flatten_axis_resources(sharding_option, in_tree,
                                                  out_tree)
    if parsed_resources is None:
        return None, None
    assert parsed_resources.mesh_axis_names is not None
    mesh_axis_names = sharding_option.mesh_axis_names
    in_op_shardings = parsed_spec_to_opsharding(
        parsed_resources.in_parsed_pspec, in_avals, mesh_shape, mesh_axis_names)
    out_op_shardings = parsed_spec_to_opsharding(
        parsed_resources.out_parsed_pspec, out_avals, mesh_shape,
        mesh_axis_names)
    return in_op_shardings, out_op_shardings

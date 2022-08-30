"""Group small ops into layers and rematerialize at layer boundary."""
from abc import ABC, abstractmethod
from functools import partial, wraps
import logging
from typing import Callable, Union, Sequence

import numpy as np
from jax import tree_flatten, lax
from jax._src.api import _check_callable
from jax._src.api import make_jaxpr
from jax._src.tree_util import tree_unflatten
from jax.core import (Var, Jaxpr, ClosedJaxpr, DropVar, Literal, jaxpr_as_fun,
                      gensym, raise_to_shaped, get_aval)
from jax.interpreters.partial_eval import remat_call_p

from alpa.global_env import global_config
from alpa.parallel_plan import PlacementSpec
from alpa.pipeline_parallel.layer_stats import (global_invar_size,
                                                is_nontrivial, eqn_flops,
                                                heavy_count,
                                                log_layer_slicing_stats)
from alpa.pipeline_parallel.primitive_def import (pipeline_p,
                                                  mark_pipeline_jaxpreqn)
from alpa.util import (clone_jaxpr, slices_to_jaxpr, OrderedSet,
                       get_var_mapping, maybe_numba_jit, new_jaxpr_eqn)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LayerOption(ABC):
    """Options of grouping operators into layers."""

    def __init__(self):
        pass

    @abstractmethod
    def transform(self, func):
        raise NotImplementedError()


class ManualLayerOption(LayerOption):
    """
    Manually specifying the boundaries of layers by using
    alpa.mark_pipeline_boundary()

    Args:
      remat_layer: Whether to use gradient rematerialization for each layer.
      static_argnums: The indices of static arguments of the
        forward function.
    """

    def __init__(self,
                 remat_layer: bool = False,
                 static_argnums: Sequence[int] = ()):
        self.remat_layer = remat_layer
        self.static_argnums = static_argnums
        super().__init__()

    def transform(self, func):
        return manual_layer_construction(func,
                                         static_argnums=self.static_argnums,
                                         remat_layer=self.remat_layer)


class AutoLayerOption(LayerOption):
    """
    Use an algorithm to automatically group operators into
    layers. The parameter `layer_num` specifies the number of
    resulting layers. You can try a few values for this parameters.
    The best choice of this value depends on the number of nodes in your
    cluster and the number of repetitive blocks in your model.

    Args:
      layer_num: The number of layers to construct.
      remat_layer: Whether to use gradient rematerialization for each layer.
      static_argnums: The indices of static arguments of the
        forward function.
    """

    def __init__(self,
                 layer_num: int,
                 remat_layer: bool = False,
                 static_argnums: Sequence[int] = ()):
        super().__init__()
        self.layer_num = layer_num
        self.remat_layer = remat_layer
        self.static_argnums = static_argnums

    def transform(self, func):
        return automatic_layer_construction(func,
                                            static_argnums=self.static_argnums,
                                            layer_num=self.layer_num,
                                            remat_layer=self.remat_layer)


class FollowLayerOption(LayerOption):
    """Follow given input placement specs to construct the layer.

    Args:
      input_placement_specs: The flatten placement specs of inputs.
      static_argnums: The indices of static arguments of the
        forward function.
    """

    def __init__(self,
                 input_placement_specs: Sequence[PlacementSpec],
                 num_meshes: int,
                 static_argnums: Sequence[int] = ()):
        super().__init__()
        self.placement_specs = input_placement_specs
        self.num_meshes = num_meshes
        self.static_argnums = static_argnums

    def transform(self, func):
        return follow_layer_construction(func, self.static_argnums,
                                         self.placement_specs, self.num_meshes)


LAYER_HEAVY_OP_LOWER_BOUND = 3
DEFAULT_EPS = 0.6
DEFAULT_COST_CRITERIA = "flops"


def slice_eqns_by_layer_boundary(closed_jaxpr: ClosedJaxpr):
    """Slices eqns by layer boundary markers."""
    sliced_eqns = []
    current_computation_eqns = []

    for eqn in closed_jaxpr.jaxpr.eqns:
        if (eqn.primitive is pipeline_p and
                eqn.params["mark_type"] == "boundary"):
            sliced_eqns.append(current_computation_eqns)
            current_computation_eqns = []
        else:
            current_computation_eqns.append(eqn)
    sliced_eqns.append(current_computation_eqns)
    return sliced_eqns


def add_pipeline_marks_for_sliced_eqns(closed_jaxpr: ClosedJaxpr, sliced_eqns):
    """Adds pipeline marks for sliced equations."""
    layer_num = len(sliced_eqns)
    layer_pipeline_invars = [OrderedSet() for _ in range(layer_num)]
    layer_pipeline_outvars = [OrderedSet() for _ in range(layer_num)]
    var_layer_dict = {}
    var_mapping = {}

    # build mapping dicts for global invars
    for var in closed_jaxpr.jaxpr.invars:
        var_layer_dict[var] = -1

    # build mapping dicts for all eqns
    for i, eqns in enumerate(sliced_eqns):
        for eqn in eqns:
            for var in eqn.invars:
                if (not isinstance(var, Literal) and
                        var not in closed_jaxpr.jaxpr.constvars and
                        var_layer_dict[var] != i):
                    layer_pipeline_invars[i].add(var)
                    if var_layer_dict[var] == -1:
                        continue
                    layer_pipeline_outvars[var_layer_dict[var]].add(var)
            for var in eqn.outvars:
                if not isinstance(var, DropVar):
                    var_layer_dict[var] = i

    # build mapping dict for global outvars
    gensym_func = gensym([closed_jaxpr.jaxpr])
    literal_outvar_eqns = []
    literal_outvar_marker_invars = []
    literal_outvar_marker_outvars = []
    for idx, var in enumerate(closed_jaxpr.jaxpr.outvars):
        if isinstance(var, Literal):
            # add a dummy equation to transform a Literal into a normal Var
            zero_literal = Literal(0, raise_to_shaped(get_aval(0)))
            new_var = gensym_func(var.aval)
            new_eqn = new_jaxpr_eqn([var, zero_literal], [new_var], lax.add_p,
                                    {})
            literal_outvar_eqns.append(new_eqn)
            literal_outvar_marker_invars.append(new_var)
            literal_outvar_marker_outvars.append(gensym_func(var.aval))
            var_mapping[idx] = literal_outvar_marker_outvars[-1]
        elif var in closed_jaxpr.jaxpr.constvars or var_layer_dict[var] == -1:
            raise NotImplementedError(
                "Does not support this use case of output var.")
        else:
            layer_pipeline_outvars[var_layer_dict[var]].add(var)

    # build new equations
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
                                   pipeline_start_outvars, f"layer_{i}",
                                   "start"))
        # all other eqns
        for eqn in (eqns + literal_outvar_eqns if i == 0 else eqns):
            new_invars = [
                get_var_mapping(computation_var_mapping, var)
                for var in eqn.invars
            ]
            new_eqns.append(
                new_jaxpr_eqn(new_invars, eqn.outvars, eqn.primitive,
                              eqn.params, eqn.effects, eqn.source_info))

        # pipeline end eqn
        pipeline_end_invars = list(
            literal_outvar_marker_invars) if i == 0 else []
        pipeline_end_outvars = list(
            literal_outvar_marker_outvars) if i == 0 else []
        for var in layer_pipeline_outvars[i]:
            new_var = gensym_func(var.aval)
            pipeline_end_invars.append(
                get_var_mapping(computation_var_mapping, var))
            pipeline_end_outvars.append(new_var)
            var_mapping[var] = new_var
        new_eqns.append(
            mark_pipeline_jaxpreqn(pipeline_end_invars, pipeline_end_outvars,
                                   f"layer_{i}", "end"))

    new_outvars = []
    for idx, var in enumerate(closed_jaxpr.jaxpr.outvars):
        if isinstance(var, Literal):
            new_outvars.append(var_mapping[idx])
        else:
            new_outvars.append(get_var_mapping(var_mapping, var))

    new_closed_jaxpr = clone_jaxpr(closed_jaxpr,
                                   outvars=new_outvars,
                                   eqns=new_eqns)
    return new_closed_jaxpr


def remat_sliced_eqns(origin_jaxpr, sliced_eqns):
    """Add tensor rematerialization for sliced equations."""
    ret_eqns = []

    sliced_jaxprs = slices_to_jaxpr(origin_jaxpr, sliced_eqns)
    for i, jaxpr in enumerate(sliced_jaxprs):
        new_invars = jaxpr.jaxpr.invars + jaxpr.jaxpr.constvars
        new_jaxpr = Jaxpr([], new_invars, jaxpr.jaxpr.outvars, jaxpr.jaxpr.eqns)
        ret_eqns.append([
            new_jaxpr_eqn(
                new_invars, new_jaxpr.outvars, remat_call_p,
                dict(concrete=False,
                     differentiated=False,
                     name=str(i),
                     call_jaxpr=new_jaxpr,
                     prevent_cse=True,
                     policy=None))
        ])
    return ret_eqns


def jaxpr_eqns_input_sizes(jaxpr) -> np.ndarray:
    """Return a list of input sizes for each equation in the jaxpr.

    Args:
        jaxpr: Jaxpr to get input sizes for.

    Returns:
        A #eqns * #eqns numpy array of input sizes. cost[l, r] represents the
        input size of the l-th to (r - 1)-th equation in the jaxpr.
    """
    length = len(jaxpr.eqns)
    input_sizes = np.full((length + 1, length + 1), 0, dtype=np.float32)

    outvars = OrderedSet()
    for k in range(0, length + 1):
        if k > 0:
            outvars = outvars.union(jaxpr.eqns[k - 1].outvars)
        invars = OrderedSet()
        total_size = 0
        for r in range(k + 1, length + 1):
            for invar in jaxpr.eqns[r - 1].invars:
                if (isinstance(invar, Var) and invar in outvars and
                        invar not in invars):
                    invars.add(invar)
                    total_size += invar.aval.size * invar.aval.dtype.itemsize
            input_sizes[k, r] = total_size
    return input_sizes


def get_layer_construction_costs(jaxpr, cost_criteria="flops"):
    """Gets the layer construction cost."""
    nontrivial = np.array([is_nontrivial(eqn) for eqn in jaxpr.eqns],
                          dtype=np.int32)
    input_sizes = jaxpr_eqns_input_sizes(jaxpr)
    if cost_criteria == "flops":
        compute_costs = np.array([
            eqn_flops(eqn) if nt else 0
            for nt, eqn in zip(nontrivial, jaxpr.eqns)
        ],
                                 dtype=np.float64)
    elif cost_criteria == "count":
        compute_costs = np.array([
            heavy_count(eqn) if nt else 0
            for nt, eqn in zip(nontrivial, jaxpr.eqns)
        ],
                                 dtype=np.float64)
    elif cost_criteria == "input_memory":
        cost_fn = partial(global_invar_size, set(jaxpr.jaxpr.invars))
        compute_costs = np.array([cost_fn(eqn) for eqn in jaxpr.eqns],
                                 dtype=np.float64)
    else:
        raise ValueError(f"Unrecoginzed cost criteria {cost_criteria}")
    return nontrivial, input_sizes, compute_costs


def cluster_jaxpr_by_cost(jaxpr: Jaxpr, layer_num: int, eps: float, costs,
                          cost_criteria):
    """Clusters the jaxpr by cost."""
    layer_num = int(layer_num)
    length = len(jaxpr.eqns)
    non_trivial, input_sizes, compute_costs = costs
    compute_costs_avg = compute_costs.sum() / layer_num
    if cost_criteria in ("flops", "input_memory"):
        compute_costs_bound = compute_costs_avg * (1 + eps)
    elif cost_criteria == "count":
        compute_costs_bound = max(compute_costs_avg * (1 + eps),
                                  compute_costs_avg + 5)
    else:
        raise ValueError(f"Unrecoginzed cost criteria {cost_criteria}")
    layer_heavy_op_lower_bound = LAYER_HEAVY_OP_LOWER_BOUND
    if sum(non_trivial) / layer_num < layer_heavy_op_lower_bound:
        layer_heavy_op_lower_bound = int(sum(non_trivial) / layer_num)  # noqa
        logger.warning(
            "Too few non-trivial ops (dot, conv), which may influence"
            " auto-sharding performance")

    @maybe_numba_jit
    def init():
        blocked = np.full((length + 1, length + 1), np.inf, dtype=np.float32)
        for left in range(1, length + 1):
            cnt = 0
            total_compute_cost = 0
            for r in range(left, length + 1):
                if non_trivial[r - 1]:
                    cnt += 1
                    total_compute_cost += compute_costs[r - 1]
                if cnt < layer_heavy_op_lower_bound:
                    if total_compute_cost >= compute_costs_bound:
                        blocked[left, r] = 0
                    continue
                if (total_compute_cost >= compute_costs_bound and
                        non_trivial[r - 1] and
                        cnt > layer_heavy_op_lower_bound):
                    break
                blocked[left, r] = 0
        return blocked

    @maybe_numba_jit
    def dp(input_sizes, blocked):
        max_cost = np.full((length + 1, layer_num + 1),
                           np.inf,
                           dtype=np.float32)
        sum_cost_under_max = np.full((length + 1, layer_num + 1),
                                     np.inf,
                                     dtype=np.float32)
        max_cost_argmin = np.full((length + 1, layer_num + 1),
                                  -1,
                                  dtype=np.int32)
        solution_imbalance = np.full((length + 1, layer_num + 1),
                                     np.inf,
                                     dtype=np.float32)
        max_cost[0, 0] = 0
        sum_cost_under_max[0, 0] = 0
        # Currently use variance to measure imbalance
        for r in range(0, length + 1):
            solution_imbalance[r, 0] = 0

        for q in range(1, layer_num + 1):
            for r in range(1, length + 1):
                for k in range(0, r):
                    new_value = max(max_cost[k, q - 1],
                                    blocked[k + 1, r] + input_sizes[k, r])
                    new_sum = (sum_cost_under_max[k, q - 1] +
                               blocked[k + 1, r] + input_sizes[k, r])
                    new_imbalance = (solution_imbalance[k, q - 1] + k**2 / q -
                                     r**2 / (q + 1) + (r - k)**2)
                    if (new_value < max_cost[r, q] or
                        (new_value <= max_cost[r, q] * (1 + 1e-4) and
                         (new_sum < sum_cost_under_max[r, q] or
                          (new_sum <= sum_cost_under_max[r, q] * (1 + 1e-4) and
                           new_imbalance < solution_imbalance[r, q])))):
                        max_cost[r, q] = new_value
                        sum_cost_under_max[r, q] = new_sum
                        max_cost_argmin[r, q] = k
                        solution_imbalance[r, q] = new_imbalance
        return max_cost_argmin, max_cost[length, layer_num]

    blocked = init()
    a_argmin, value = dp(input_sizes, blocked)

    reversed_sliced_eqns = []

    r = length
    for q in range(layer_num, 0, -1):
        k = a_argmin[r, q]
        reversed_sliced_eqns.append(jaxpr.eqns[k:r])
        r = k
    assert r == 0, "No solution for layer construction."
    solution = list(reversed(reversed_sliced_eqns))

    # print("dp solution")
    # for i, eqns in enumerate(solution):
    #    invars = OrderedSet()
    #    for eqn in eqns:
    #        invars.update([var for var in eqn.invars if isinstance(var, Var)])
    #    invars.intersection_update(jaxpr.jaxpr.invars)
    #    print(f"mesh: {i},  set_shapes: "
    #          f"{[x.aval.shape for x in invars if len(x.aval.shape) > 1]}")
    #
    #    invars = []
    #    for eqn in eqns:
    #        tmp_set = set([var for var in eqn.invars if isinstance(var, Var)])
    #        tmp_set.intersection_update(jaxpr.jaxpr.invars)
    #        invars.extend(list(tmp_set))
    #    print(f"mesh: {i}, list_shapes: "
    #          f"{[x.aval.shape for x in invars if len(x.aval.shape) > 1]}")

    solution_info = {
        "total_cost": value,
    }
    return solution, solution_info


def search_layer_num(jaxpr,
                     eps,
                     layer_eps=0,
                     cost_criteria=DEFAULT_COST_CRITERIA):
    """TODO(zhuohan): docstring."""
    non_trivial, input_sizes, compute_costs = get_layer_construction_costs(
        jaxpr)
    layer_num = 2
    r = int(non_trivial.sum() / 3) + 1
    _, solution_info = cluster_jaxpr_by_cost(
        jaxpr,
        layer_num,
        eps, (non_trivial, input_sizes, compute_costs),
        cost_criteria=cost_criteria)
    l_val = solution_info["total_cost"]
    while r - layer_num > 1:
        mid = int((layer_num + r) / 2)
        _, solution_info = cluster_jaxpr_by_cost(
            jaxpr,
            mid,
            eps, (non_trivial, input_sizes, compute_costs),
            cost_criteria=cost_criteria)
        mid_val = solution_info["total_cost"]
        if mid_val > l_val * (1 + layer_eps):
            r = mid
        else:
            layer_num = mid
    return layer_num


def layer_level_jaxpr_transformation(fn: Callable,
                                     static_argnums: Sequence[int] = (),
                                     remat: bool = False,
                                     layer_construction: bool = False,
                                     auto_layer_boundary: bool = False,
                                     layer_num: Union[int, str] = None,
                                     eps: float = DEFAULT_EPS,
                                     cost_criteria: str = DEFAULT_COST_CRITERIA,
                                     layer_eps: float = 0.0):
    """TODO(zhuohan): docstring."""
    if not remat and not layer_construction:
        return fn

    @wraps(fn)
    def wrapped(*args):
        jaxpr, out_shape_tree = make_jaxpr(fn,
                                           static_argnums=static_argnums,
                                           return_shape=True)(*args)
        if auto_layer_boundary:
            nonlocal layer_num
            if layer_num == "auto":
                layer_num = search_layer_num(jaxpr, eps, layer_eps)
            costs = get_layer_construction_costs(jaxpr,
                                                 cost_criteria=cost_criteria)
            sliced_eqns, _ = cluster_jaxpr_by_cost(jaxpr,
                                                   layer_num,
                                                   eps,
                                                   costs,
                                                   cost_criteria=cost_criteria)
            if global_config.print_auto_layer_stats:
                log_layer_slicing_stats(jaxpr, sliced_eqns)
        else:
            sliced_eqns = slice_eqns_by_layer_boundary(jaxpr)

        if remat:
            sliced_eqns = remat_sliced_eqns(jaxpr, sliced_eqns)

        if layer_construction:
            jaxpr = add_pipeline_marks_for_sliced_eqns(jaxpr, sliced_eqns)
        else:
            jaxpr = clone_jaxpr(jaxpr,
                                eqns=[x for eqns in sliced_eqns for x in eqns])

        flatten_args, _ = tree_flatten(args)
        ans = jaxpr_as_fun(jaxpr)(*flatten_args)  # pylint: disable=not-callable
        _, out_tree = tree_flatten(out_shape_tree)
        return tree_unflatten(out_tree, ans)

    return wrapped


def manual_remat(fun: Callable = None, *, static_argnums: Sequence[int] = ()):
    """Rematerialize an input function with manually selected layer boundaries.

    Rematerialize each layer of an input function with manually selected layer
    boundaries indicated by pipeline markers.

    Args:
        fun: the input function to rematerialize.
        static_argnums: An optional int or collection of ints that specify
          which positional arguments to treat as static (compile-time constant).
          Same as in jax.jit
    Returns:
        A new function rematerializes each layer of the input function.
    """

    def decorate_fun(fun):
        return layer_level_jaxpr_transformation(fun,
                                                static_argnums,
                                                remat=True,
                                                layer_construction=False,
                                                auto_layer_boundary=False)

    if fun is None:
        return decorate_fun
    else:
        _check_callable(fun)
        return decorate_fun(fun)


def automatic_remat(fun: Callable = None,
                    *,
                    static_argnums: Sequence[int] = (),
                    layer_num: Union[int, str] = None,
                    eps: float = DEFAULT_EPS,
                    cost_criteria: str = DEFAULT_COST_CRITERIA,
                    layer_eps: float = 0.0):
    """Rematerialize an input function with automatic boundaries.

    Rematerialize each layer of an input function with automatically decided
    layer boundaries.

    Args:
        fun: The input function to rematerialize.
        static_argnums: An optional int or collection of ints that specify
          which positional arguments to treat as static (compile-time constant).
          Same as in jax.jit
        layer_num: The number of layers to rematerialize. If set to "auto", the
          number of layers will be automatically determined by a binary search.
          The binary search might not work for complex input functions.
        eps: The tolerance of inbalance of the costs of different layers.
        cost_criteria: The cost criteria to use for deciding the layers.
        layer_eps: A parameter for layer_num binary search.

    Returns:
        A new function rematerializes each layer of the input function.
    """

    def decorate_fun(fun):
        return layer_level_jaxpr_transformation(fun,
                                                static_argnums,
                                                remat=True,
                                                layer_construction=False,
                                                auto_layer_boundary=True,
                                                layer_num=layer_num,
                                                eps=eps,
                                                cost_criteria=cost_criteria,
                                                layer_eps=layer_eps)

    if fun is None:
        return decorate_fun
    else:
        _check_callable(fun)
        return decorate_fun(fun)


def manual_layer_construction(fun: Callable = None,
                              *,
                              static_argnums: Sequence[int] = (),
                              remat_layer: bool = False):
    """Setup manually selected layer boundaries.

    Add input variables of each layer to its start pipeline marker and output
    variables of each layer to its end pipeline marker.

    Args:
        fun: the input function.
        static_argnums: An optional int or collection of ints that specify
          which positional arguments to treat as static (compile-time constant).
          Same as in jax.jit
        remat_layer: Whether to rematerialize each layer at layer boundaries.
    Returns:
        A new function with correctly setup pipeline markers.
    """

    def decorate_fun(fun):
        return layer_level_jaxpr_transformation(fun,
                                                static_argnums,
                                                remat=remat_layer,
                                                layer_construction=True,
                                                auto_layer_boundary=False)

    if fun is None:
        return decorate_fun
    else:
        _check_callable(fun)
        return decorate_fun(fun)


def automatic_layer_construction(fun: Callable = None,
                                 *,
                                 static_argnums: Sequence[int] = (),
                                 layer_num: int = None,
                                 remat_layer: bool = False,
                                 eps: float = DEFAULT_EPS,
                                 cost_criteria: str = DEFAULT_COST_CRITERIA,
                                 layer_eps: float = 0.0):
    """Automatically cluster the equations in a jaxpr into layers.
    Automatically cluster the equations in a jaxpr into layers and add pipeline
    markers at layer boundaries.
    Args:
        fun: the input function.
        static_argnums: An optional int or collection of ints that specify
          which positional arguments to treat as static (compile-time constant).
          Same as in jax.jit
        layer_num: the number of layers to rematerialize. If set to "auto", the
          number of layers will be automatically determined by a binary search.
          The binary search might not work for complex input functions.
        remat_layer: Whether to rematerialize each layer at layer boundaries.
        eps: the tolerance of inbalance of the costs of different layers.
        cost_criteria: the cost criteria to use for deciding the layers.
        layer_eps: a parameter for layer_num binary search.
    Returns:
        A new function rematerializes each layer of the input function.
    """

    def decorate_fun(fun):
        return layer_level_jaxpr_transformation(fun,
                                                static_argnums,
                                                remat=remat_layer,
                                                layer_construction=True,
                                                auto_layer_boundary=True,
                                                layer_num=layer_num,
                                                eps=eps,
                                                cost_criteria=cost_criteria,
                                                layer_eps=layer_eps)

    if fun is None:
        return decorate_fun
    else:
        _check_callable(fun)
        return decorate_fun(fun)


def follow_layer_construction(fun, static_argnums, input_placement_specs,
                              num_meshes):
    """Follow given input placement specs to construct layers."""
    _check_callable(fun)

    @wraps(fun)
    def wrapped(*args):
        jaxpr, out_shape_tree = make_jaxpr(fun,
                                           static_argnums=static_argnums,
                                           return_shape=True)(*args)

        var2mesh = {}  # Dict[var -> mesh_idx]

        for var, spec in zip(jaxpr.jaxpr.invars, input_placement_specs):
            if spec is None:
                # Assign input vars to mesh 0 by default
                if isinstance(var, Var):
                    var2mesh[var] = 0
            else:
                if isinstance(var, Var):
                    var2mesh[var] = spec.mesh_ids[0]

        sliced_eqns = slice_jaxpr_with_var_assignment(jaxpr, var2mesh,
                                                      num_meshes)
        jaxpr = add_pipeline_marks_for_sliced_eqns(jaxpr, sliced_eqns)

        flatten_args, _ = tree_flatten(args)
        ans = jaxpr_as_fun(jaxpr)(*flatten_args)  # pylint: disable=not-callable
        _, out_tree = tree_flatten(out_shape_tree)
        return tree_unflatten(out_tree, ans)

    return wrapped


def slice_jaxpr_with_var_assignment(jaxpr, var2mesh, num_meshes):
    mesh_begin = [None] * num_meshes
    mesh_end = [None] * num_meshes

    # Run a linear scan to find the begin and end equations of each mesh.
    cur_mesh = 0
    for idx, eqn in enumerate(jaxpr.eqns):
        if eqn.primitive is pipeline_p:
            raise ValueError("FollowParallel is not compatible with manual "
                             "pipeline marker. Please do not insert manual "
                             "pipeline marker in the function.")
        for var in eqn.invars:
            if isinstance(var, Var) and var in var2mesh:
                mesh_idx = var2mesh[var]

                if mesh_idx > cur_mesh:
                    cur_mesh = mesh_idx

                if mesh_begin[cur_mesh] is None:
                    mesh_begin[cur_mesh] = idx
                mesh_end[cur_mesh] = idx

    # Some boundary equations are not within the ranges detected above.
    # Use DP algorithm to refine the boundary, so we can minimize the
    # communication costs.
    cost_criteria = "flops"
    costs = get_layer_construction_costs(jaxpr, cost_criteria=cost_criteria)
    _, _, compute_costs = costs

    # To make the solution of DP algorithm respect our begin/end constraint.
    # We assign begin, end equations a very large cost and run DP
    # with a small eps.
    max_cost = np.sum(compute_costs) * 10
    for i in range(num_meshes):
        assert mesh_begin[i] is not None and mesh_end[i] is not None
        compute_costs[mesh_begin[i]] += max_cost
        compute_costs[mesh_end[i]] += max_cost

    sliced_eqns, _ = cluster_jaxpr_by_cost(jaxpr,
                                           layer_num=num_meshes,
                                           eps=0.1,
                                           costs=costs,
                                           cost_criteria=cost_criteria)
    return sliced_eqns

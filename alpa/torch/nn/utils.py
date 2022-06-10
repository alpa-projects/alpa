# pylint: skip-file

# All code in this file are extracted from torchdynamo and functorch.
# Skipping pylint for this file so that it's easy to find out the difference
# when we need to pull in new changes again.

import builtins
import dataclasses
import functools
import itertools
import math
import operator
from typing import List

import torch
from torch import nn
from torch import Tensor
from torch.fx import Transformer
from torch.fx.experimental.normalize import NormalizeOperators
from torch.fx.operator_schemas import get_signature_for_torch_op

# Copied from torchdynamo/torchdynamo/optimizations/normalize.py
VIEW_OPS = {
    # list taken from https://pytorch.org/docs/stable/tensor_view.html
    "getitem",
    "as_strided",
    "detach",
    "diagonal",
    "expand",
    "expand_as",
    "movedim",
    "narrow",
    "permute",
    "select",
    "squeeze",
    "transpose",
    "t",
    "T",
    "real",
    "imag",
    "view_as_real",
    "view_as_imag",
    "unflatten",
    "unfold",
    "unsqueeze",
    "view",
    "view_as",
    "unbind",
    "split",
    "split_with_sizes",
    "swapaxes",
    "swapdims",
    "chunk",
    "indices",
    "values",
}
MAYBE_VIEW_OPS = {"contiguous", "reshape"}

# convert x.foo(...) to torch.foo(x, ...)
NORMALIZE_METHODS = {
    # These ones aren't normalized:
    # ('view', 342)
    # ('reshape', 285)
    # ('expand', 87)
    # ('permute', 78)
    # ('to', 66)
    # ('contiguous', 62)
    # ('reshape_as', 57)
    # ('masked_fill', 30)
    # ('float', 22) -- could rewrite
    # ('expand_as', 14) -- could rewrite
    # ('detach', 4)
    # ('repeat', 2)
    # TODO(jansel): debug why this causes issues in detectron2_maskrcnn
    # "div": torch.div,
    "add_": operator.iadd,
    "all": torch.all,
    "any": torch.any,
    "ceil": torch.ceil,
    "chunk": torch.chunk,
    "clamp": torch.clamp,
    "clone": torch.clone,
    "exp": torch.exp,
    "flatten": torch.flatten,
    "flip": torch.flip,
    "floor": torch.floor,
    "index_select": torch.index_select,
    "log2": torch.log2,
    "log_softmax": torch.nn.functional.log_softmax,
    "max": torch.max,
    "mean": torch.mean,
    "min": torch.min,
    "mul_": operator.imul,
    "narrow": torch.narrow,
    "ne": torch.ne,
    "nonzero": torch.nonzero,
    "numel": torch.numel,
    "pow": torch.pow,
    "round": torch.round,
    "rsqrt": torch.rsqrt,
    "sigmoid": torch.sigmoid,
    "softmax": torch.nn.functional.softmax,
    "sort": torch.sort,
    "split": torch.split,
    "squeeze": torch.squeeze,
    "std": torch.std,
    "sum": torch.sum,
    "topk": torch.topk,
    "transpose": torch.transpose,
    "tril": torch.tril,
    "t": torch.t,
    "unbind": torch.unbind,
    "unsqueeze": torch.unsqueeze,
}
DONT_EXPAND_MODULES = {
    # These have internal control flow
    "ConvTranspose1d",
    "ConvTranspose2d",
    "Conv2d",
    "ConvReLU2d",
    "ConvBn2d",
    "ConvBnReLU2d",
    "EmbeddingBag",
    "InstanceNorm2d",
    "LSTM",
}

F = torch.nn.functional
INPLACE_KEYWORD_OPS = {
    F.mish,
    F.silu,
    F.hardsigmoid,
    F.rrelu,
    F.leaky_relu,
    F.celu,
    F.selu,
    F.elu,
    F.relu6,
    F.hardswish,
    F.hardtanh,
    F.relu,
    F.threshold,
}
IOPERATOR_REPLACEMENTS = {
    "masked_fill_": "masked_fill",
    "scatter_": "scatter",
    "unsqueeze_": "unsqueeze",
    torch.relu_: torch.relu,
    torch.sigmoid_: torch.sigmoid,
    operator.iadd: torch.add,
    operator.iand: torch.bitwise_and,
    operator.ifloordiv: functools.partial(torch.div, rounding_mode="floor"),
    operator.itruediv: torch.div,
    operator.imul: torch.mul,
    operator.imatmul: torch.matmul,
    operator.ior: torch.bitwise_or,
    operator.ipow: torch.pow,
    operator.isub: torch.sub,
    operator.ixor: torch.bitwise_xor,
}
OPERATOR_REPLACEMENTS = {
    operator.lt:
        torch.lt,
    operator.le:
        torch.le,
    operator.eq:
        torch.eq,
    operator.ne:
        torch.ne,
    operator.ge:
        torch.ge,
    operator.gt:
        torch.gt,
    operator.abs:
        torch.abs,
    operator.add:
        torch.add,
    operator.and_:
        torch.bitwise_and,
    operator.floordiv:
        functools.partial(torch.div, rounding_mode="floor"),
    # operator.truediv: torch.div,  # TODO(jansel): debug issue in vision_maskrcnn
    operator.inv:
        torch.bitwise_not,
    operator.invert:
        torch.bitwise_not,
    operator.mod:
        torch.remainder,
    operator.mul:
        torch.mul,
    operator.matmul:
        torch.matmul,
    operator.neg:
        torch.neg,
    operator.or_:
        torch.bitwise_or,
    operator.pos:
        torch.positive,
    operator.pow:
        torch.pow,
    operator.sub:
        torch.sub,
    operator.xor:
        torch.bitwise_xor,
    torch.nn.functional.sigmoid:
        torch.sigmoid,
    torch.nn.functional.tanh:
        torch.tanh,
    torch.nn.functional.relu:
        torch.relu,
}

SKIP_INPLACE = {
    v for v in itertools.chain(math.__dict__.values(), builtins.__dict__.values(
    ), operator.__dict__.values()) if callable(v)
}


def always_true(*args, **kwargs):
    return True


class InliningTracer(torch.fx.Tracer):

    def is_leaf_module(self, m: torch.nn.Module,
                       module_qualified_name: str) -> bool:
        return False


def expand_module_call(prefix, graph: torch.fx.Graph, module, args, kwargs):
    # this patch is needed to make BatchNorm2D FX trace
    module.__dict__["_check_input_dim"] = always_true
    try:
        assert not kwargs
        arg_index = itertools.count()
        vars = dict()
        for node in InliningTracer().trace(module).nodes:
            if node.op == "placeholder":
                vars[node] = args[next(arg_index)]
            elif node.op == "output":
                assert len(node.args) == 1
                return vars[node.args[0]]
            elif node.op == "get_attr":
                vars[node] = graph.get_attr(f"{prefix}{node.target}")
            else:
                vars[node] = graph.node_copy(node, vars.__getitem__)
        assert False
    except Exception:
        print(f"Error while expanding {module.__class__.__name__}")
        raise
    finally:
        del module.__dict__["_check_input_dim"]


@dataclasses.dataclass
class NodeCounts:
    usages: int = 0


def short_name(gm, node: torch.fx.Node):
    if node.op == "call_function":
        return node.target.__name__
    elif node.op == "call_method":
        return node.target
    elif node.op == "call_module":
        return gm.get_submodule(node.target).__class__.__name__
    elif node.op == "get_attr":
        return node.target
    elif node.op == "output":
        return "output"
    assert False, node.op


def long_name(gm, node: torch.fx.Node):
    name = short_name(gm, node)
    target = node.target
    if node.op == "call_function":
        return torch_get_name(node.target,
                              f"{getattr(target, '__module__', '')}.{name}")
    elif node.op == "call_method":
        return name
    elif node.op == "call_module":
        target = gm.get_submodule(target).__class__
        return f"{getattr(target, '__module__', '')}.{getattr(target, '__name__', '')}"
    elif node.op == "get_attr":
        return name
    elif node.op == "output":
        return "output"
    assert False


class Inplacifier:

    def __init__(self, gm: torch.fx.GraphModule):
        self.gm = gm

    def can_be_view(self, node):
        name = short_name(self.gm, node)
        return name in VIEW_OPS or name in MAYBE_VIEW_OPS

    def inplacify(self):
        counts = dict()

        def record_usage(node):
            counts[node].usages += 1
            return node

        for node in self.gm.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                if self.can_be_view(node):
                    # Aliasing
                    counts[node] = counts[node.args[0]]
                elif "out" in node.kwargs:
                    counts[node] = counts[node.kwargs["out"]]
                else:
                    counts[node] = NodeCounts(0)
            else:
                counts[node] = NodeCounts(float("inf"))

        for node in reversed(list(self.gm.graph.nodes)):
            kwargs = dict(node.kwargs)
            if "inplace" in kwargs:
                kwargs.pop("inplace")
            if node.op == "call_function" and len(node.args) + len(kwargs) == 1:
                arg = node.args[0] if node.args else next(kwargs.values())
                if isinstance(arg, torch.fx.Node) and counts[arg].usages == 0:
                    if node.target in SKIP_INPLACE:
                        continue
                    elif node.target in INPLACE_KEYWORD_OPS:
                        kwargs["inplace"] = True
                        counters["optimizations"]["inplace"] += 1
                    elif " out: torch.Tensor" in repr(
                            get_signature_for_torch_op(node.target)):
                        kwargs["out"] = arg
                        counters["optimizations"]["out"] += 1
                    else:
                        continue
                    with self.gm.graph.inserting_before(node):
                        node.replace_all_uses_with(
                            self.gm.graph.call_function(node.target, node.args,
                                                        kwargs))
                    self.gm.graph.erase_node(node)

            torch.fx.map_arg((node.args, node.kwargs), record_usage)


class Functionalization(Transformer):
    """Remove most cases of mutation from a given fx Graph.
    """

    def __init__(self, *args, **kwargs):
        super(Functionalization, self).__init__(*args, **kwargs)
        self.tracer.tensor_attrs = dict()  # TODO(jansel): upstream this fix

    def run_node(self, n: torch.fx.Node):

        patches = []
        target = n.target
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        kwargs = dict(kwargs)

        if (not n.meta["is_input_mutation"] and
                not n.meta["partial_mutation"] and
                issubclass(n.meta["type"], torch.Tensor)):
            if "inplace" in n.kwargs:
                if kwargs["inplace"]:
                    patches.append(n.args[0])
                kwargs.pop("inplace")
            elif "out" in n.kwargs:
                kwargs.pop("out")
                patches.append(n.kwargs["out"])
            elif n.target in IOPERATOR_REPLACEMENTS:
                target = IOPERATOR_REPLACEMENTS[n.target]
                patches.append(n.args[0])
            elif n.meta["is_mutation"]:
                counters["mutation"][long_name(self.module, n)] += 1

            if target in OPERATOR_REPLACEMENTS and not kwargs:
                target = OPERATOR_REPLACEMENTS[target]

        if target is builtins.getattr:
            if args[1] == "dtype":
                return n.args[0].meta["dtype"]
            elif args[1] == "device":
                return n.args[0].meta["device"]
            else:
                counters["getattr"][args[1]] += 1

        if isinstance(target, functools.partial):
            assert not target.args
            kwargs.update(target.keywords)
            target = target.func

        if not issubclass(n.meta["type"], torch.Tensor):
            counters["nontensor"][long_name(self.module, n)] += 1

        result = getattr(self, n.op)(target, args, kwargs)

        for patch in patches:
            assert isinstance(
                patch, torch.fx.Node), f"{patch} {n.target} {n.args} {n.kwargs}"
            if patch in self.env:
                self.env[patch] = result

        return result


def swap_node(graph, old_node, new_node):
    old_node.replace_all_uses_with(new_node)
    graph.erase_node(old_node)


def normalize(gm: torch.fx.GraphModule):
    # gm.graph.print_tabular()
    graph: torch.fx.Graph = gm.graph

    for node in list(graph.nodes):
        with graph.inserting_before(node):
            if node.op == "call_method" and node.target in NORMALIZE_METHODS:
                swap_node(
                    graph,
                    node,
                    graph.call_function(NORMALIZE_METHODS[node.target],
                                        node.args, node.kwargs),
                )
            elif node.op == "call_module":
                submod = gm.get_submodule(node.target)
                if submod.__class__.__name__ not in DONT_EXPAND_MODULES:
                    swap_node(
                        graph,
                        node,
                        expand_module_call(f"{node.target}.", graph, submod,
                                           node.args, node.kwargs),
                    )

    # gm.graph.print_tabular()


def create_names_map(named_params, tied_named_params):
    """named_params is a dictionary of tensors: {'A': A, 'B': B}
    tied_named_params is another dictionary of tensors {'A': A, 'B': B, 'B_tied': B}
    with potentially tied (or 'duplicated') tensors

    This function creates a mapping from the names in named_params to the
    names in tied_named_params: {'A': ['A'], 'B': ['B', 'B_tied']}.
    """
    named_params = {k: v for k, v in named_params}
    tied_named_params = {k: v for k, v in tied_named_params}

    tensors_dict_keys = set(named_params.keys())
    tied_tensors_dict_keys = set(tied_named_params.keys())
    assert tensors_dict_keys.issubset(tied_tensors_dict_keys)

    tensor_to_mapping = {}
    for key, tensor in named_params.items():
        tensor_to_mapping[tensor] = (key, [])
    for key, tensor in tied_named_params.items():
        assert tensor in tensor_to_mapping
        tensor_to_mapping[tensor][1].append(key.split("."))
    result = {key: value for key, value in tensor_to_mapping.values()}
    return result


def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def _extract_members(mod: nn.Module, _named_members, named_members, subclass):
    all_named_members = tuple(_named_members(mod, remove_duplicate=False))
    named_members = tuple(named_members())
    names_map = create_names_map(named_members, all_named_members)

    # Remove all the members in the model
    memo = {}
    for name, p in all_named_members:
        if p not in memo:
            memo[p] = subclass(torch.empty_like(p, device="meta"))
        replacement = memo[p]
        _set_nested_attr(mod, name.split("."), replacement)

    if len(named_members) == 0:
        names, params = (), ()
    else:
        names, params = zip(*named_members)
    return params, names, names_map


def extract_weights(mod: nn.Module):
    """This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    return _extract_members(mod, named_parameters, mod.named_parameters,
                            nn.Parameter)


def extract_buffers(mod: nn.Module):
    return _extract_members(mod, named_buffers, mod.named_buffers, lambda x: x)


# Copied from functorch/functorch/_src/named_members_polyfill.py
def named_members(mod,
                  get_members_fn,
                  prefix='',
                  recurse=True,
                  remove_duplicate=True):
    """Helper method for yielding various names + members of modules.
    """
    memo = set()
    modules = mod.named_modules(
        prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [
            (prefix, mod)
        ]
    for module_prefix, module in modules:
        members = get_members_fn(module)
        for k, v in members:
            if v is None or v in memo:
                continue
            if remove_duplicate:
                memo.add(v)
            name = module_prefix + ('.' if module_prefix else '') + k
            yield name, v


def named_parameters(mod,
                     prefix: str = '',
                     recurse: bool = True,
                     remove_duplicate: bool = True):
    return named_members(mod,
                         lambda module: module._parameters.items(),
                         prefix=prefix,
                         recurse=recurse,
                         remove_duplicate=remove_duplicate)


def named_buffers(mod,
                  prefix: str = '',
                  recurse: bool = True,
                  remove_duplicate: bool = True):
    return named_members(mod,
                         lambda module: module._buffers.items(),
                         prefix=prefix,
                         recurse=recurse,
                         remove_duplicate=remove_duplicate)

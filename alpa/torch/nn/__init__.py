"""PyTorch module conversion related functions.
"""
import copy
from typing import List, Callable, Dict
from collections import OrderedDict

import torch
from torch import Tensor, nn
from torch.fx.experimental.normalize import NormalizeOperators
from torchdistx import deferred_init as torchdistx_deferred_init
from torchdistx.fake import meta_like

import alpa.torch as atorch
from alpa.torch.tensor_utils import make_shaped_array_from_pt_tensor
from alpa.torch.nn.utils import (DONT_EXPAND_MODULES, extract_buffers,
                                 extract_weights, named_buffers, named_members,
                                 named_parameters, normalize)

mapping_prefix = "alpa_torch_ops_mapping"


def fx_ir_to_alpa_func_code(fx_ir, alpa_func_name):
    # TODO: maybe we can operate on FX IR node to clean up this impl

    fx_ir_code_cleaned = ""
    for line in fx_ir.code.strip().split("\n"):
        line = line.replace(";  ", "\n    ")
        fx_ir_code_cleaned += line + "\n"

    if atorch.debug:
        print("FX IR code (cleaned): ")
        print(fx_ir_code_cleaned)

    lines = fx_ir_code_cleaned.split("\n")
    assert "def forward(" in lines[0]
    signature_line = lines[0]
    sig_args = signature_line.split("def forward(")[1].split("):")[0].split(
        ", ")
    sig_args = sig_args[1:]  # remove `self`
    sig_args.insert(0, "params")
    sig_args.insert(1, "bufs")
    signature_line = f"def {alpa_func_name}(" + ", ".join(sig_args) + "):"

    out_body_lines = []

    bufs_set = set(fx_ir.buffers(recurse=True))
    bufs_n_to_key = {}

    for line in lines[1:]:
        line = line.replace(" : torch.Tensor", "")
        if "self." in line:
            if "getattr(" in line:
                # Example line in IR:
                # `... = getattr(self.layers, "0").encoder.self_attn.qkv.weight`
                # For RHS, FQN in param dict should be:
                # "layers.0.encoder.self_attn.qkv.weight"
                attr_fqn_name_in_original_ir = line.split(" = ")[1]
                attr_fqn_name_in_param_dict = (
                    line.split("getattr(self.")[1].split("(")[0].replace(
                        ', "', ".").replace('")', ""))
            else:
                # Example line in IR:
                # `self_layers_0__w_attention = self.self_layers_0__w_attention`
                # For RHS, FQN in param dict should be:
                # "self_layers_0__w_attention"
                attr_fqn_name_in_original_ir = line.split(" = ")[1]
                attr_fqn_name_in_param_dict = line.split("self.")[1].split(
                    "(")[0]
            line_rhs = line.split(" = ")[1]
            try:
                if ")." in line_rhs:
                    # Example line in IR:
                    # `... = getattr(self.layers, "0").conv(reshape_7)`
                    # Attribute access statement should be
                    # `getattr(self.layers, "0").conv`
                    attr_access_stmt = ("_tmp_value = " +
                                        line_rhs.split(").")[0].replace(
                                            "self.", "locals()['fx_ir'].") +
                                        ")." +
                                        line_rhs.split(").")[1].split("(")[0])
                else:
                    attr_access_stmt = "_tmp_value = " + line_rhs.replace(
                        "self.", "locals()['fx_ir'].")
            except IndexError as e:
                print(line_rhs)
                raise e
            # pylint: disable=exec-used
            exec(attr_access_stmt)
            attr_value = locals()["_tmp_value"]
            if isinstance(attr_value, torch.nn.Module):
                # Full list of NN modules that need this handling is at
                # torchdynamo/torchdynamo/optimizations/normalize.py
                # `DONT_EXPAND_MODULES`.
                assert attr_value.__class__.__name__ in DONT_EXPAND_MODULES, \
                    "unknown module: " + str(attr_value.__class__.__name__)
                call_args = line.split("self.")[1].split("(")[1].split(
                    ")")[0].split(", ")
                if attr_value.__class__.__name__ == "Conv2d":
                    call_args += [
                        f"params['{attr_fqn_name_in_param_dict}.weight']",
                        f"bias=params['{attr_fqn_name_in_param_dict}.bias']",
                        f"stride={attr_value.stride}",
                        f"padding={attr_value.padding}",
                        f"dilation={attr_value.dilation}",
                        f"groups={attr_value.groups}",
                    ]
                    lhs = line.split(" = ")[0]
                    line = lhs + " = " + f"torch.conv2d({', '.join(call_args)})"
                else:
                    raise NotImplementedError
            elif isinstance(attr_value, torch.nn.Parameter):  # Parameter
                line = line.replace(f"{attr_fqn_name_in_original_ir}",
                                    f"params['{attr_fqn_name_in_param_dict}']")
            elif isinstance(attr_value, torch.Tensor):
                if attr_value in bufs_set:  # Buffer
                    # TODO: verify whether torch.fx.symbolic_trace
                    # puts both buffer and non-buffer Tensors
                    # (i.e. both `self.register_buffer(...)` and
                    # `self.tensor = torch.tensor(...)`)
                    # into buffers dict.
                    # This code assumes so.
                    line = line.replace(
                        f"{attr_fqn_name_in_original_ir}",
                        f"bufs['{attr_fqn_name_in_param_dict}']")
                else:  # Const
                    raise ValueError(
                        "We assume torch.fx treats non-buffer "
                        "tensor attributes as buffers, "
                        "but this assumption no longer holds true for "
                        ".{attr_fqn_name_in_param_dict}")
            else:  # Const
                raise ValueError(
                    "non-module / non-tensor attribute is not supported, "
                    "but found type of "
                    f"'{attr_fqn_name_in_param_dict}' to be {type(attr_value)}")

        # Record all buffers' name and their correponding key in `bufs` dict
        if " = bufs['" in line:
            buf_name = line.split(" = bufs['")[0].strip()
            buf_key = line.split(" = bufs['")[1].split("']")[0]
            bufs_n_to_key[buf_name] = buf_key

        # Rewrite stateful modules / ops
        if "torch.nn.functional.batch_norm" in line:
            lhs = line.split(" = torch.nn.functional.batch_norm")[0]
            call_args = line.split(" = torch.nn.functional.batch_norm("
                                  )[1].split(")")[0].split(", ")
            r_mean_arg_n = call_args[1]
            assert "running_mean" in r_mean_arg_n
            r_var_arg_n = call_args[2]
            assert "running_var" in r_var_arg_n
            line = (lhs + ", r_mean_new, r_var_new" +
                    " = torch.nn.functional.batch_norm(" +
                    ", ".join(call_args) + ")")
            line += "\n"
            line += f"    bufs['{bufs_n_to_key[r_mean_arg_n]}'] = r_mean_new"
            line += "\n"
            line += f"    bufs['{bufs_n_to_key[r_var_arg_n]}'] = r_var_new"

        # Op lowering
        if "torch._C._nn." in line:
            op_name = line.split("torch._C._nn.")[1].split("(")[0]
            line = line.replace(f"torch._C._nn.{op_name}",
                                f"torch.nn.functional.{op_name}")
        if f"{mapping_prefix}_torch_nn_functional_" in line:
            op_name = line.split(
                f"{mapping_prefix}_torch_nn_functional_")[1].split("(")[0]
            line = line.replace(
                f"{mapping_prefix}_torch_nn_functional_{op_name}",
                f"torch.nn.functional.{op_name}")
        if f"{mapping_prefix}_torch_" in line:
            op_name = line.split(f"{mapping_prefix}_torch_")[1].split("(")[0]
            line = line.replace(f"{mapping_prefix}_torch_{op_name}",
                                f"torch.{op_name}")
        if ".dim()" in line:
            tensor_name = line.split(" = ")[1].split(".dim()")[0]
            line = line.replace(f"{tensor_name}.dim()",
                                f"len({tensor_name}.shape)")
        if ".size()" in line:
            tensor_name = line.split(" = ")[1].split(".size()")[0]
            line = line.replace(f"{tensor_name}.size()", f"{tensor_name}.shape")
        if ".permute(" in line:
            tensor_name = line.split(" = ")[1].split(".permute(")[0]
            line = line.replace(f"{tensor_name}.permute(",
                                f"torch.permute({tensor_name}, (") + ")"
        if ".expand(" in line:
            tensor_name = line.split(" = ")[1].split(".expand(")[0]
            line = line.replace(f"{tensor_name}.expand(",
                                f"torch.expand({tensor_name}, (") + ")"
        if ".view(" in line:
            tensor_name = line.split(" = ")[1].split(".view(")[0]
            line = line.replace(f"{tensor_name}.view(",
                                f"torch.view({tensor_name}, (") + ")"
        if " @ " in line:
            lhs = line.split(" = ")[0]
            rhs = line.split(" = ")[1]
            line = lhs + " = " + "torch.matmul(" + rhs.replace(" @ ",
                                                               ", ") + ")"

        if "return " in line:
            rhs_of_return = line.split("return ")[1]
            output_args = rhs_of_return.split(",")
            output_args.insert(0, "bufs")
            line = line.split("return ")[0] + "return " + ", ".join(output_args)

        out_body_lines.append(line)

    # `alpa_func_code` is string form of a function that contains
    # (mostly) PyTorch operations.
    # "mostly" because ops like `torch.expand` and `torch.view` are not actually
    # valid PyTorch ops and only work within `atorch.bind_ops()` context.
    alpa_func_code = signature_line + "\n" + "\n".join(out_body_lines) + "\n"
    alpa_func_code = alpa_func_code.strip()

    return alpa_func_code


# Copied from torchdynamo/torchdynamo/optimizations/normalize.py
def normalize_ir_no_run(fx_ir):
    normalize(fx_ir)
    try:
        fx_ir = NormalizeOperators(fx_ir).transform()
    except AttributeError:
        # log.exception("NormalizeOperators() failed")
        pass
    # ShapeAliasingAndMutationProp(fx_ir).run(*example_inputs)
    # fx_ir = Functionalization(fx_ir).transform()
    fx_ir.recompile()
    # record_graph_stats(fx_ir)
    return fx_ir


# Copied from functorch/functorch/_src/make_functional.py
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def _get_nested_attr(obj: nn.Module, names: List[str]) -> None:
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return _get_nested_attr(getattr(obj, names[0]), names[1:])


def _swap_state(mod: nn.Module, names_map: Dict[str, List[str]], elems):
    result = []
    for (_, attr_names), elem in zip(names_map.items(), elems):
        for i, attr_name in enumerate(attr_names):
            if i == 0:
                result.append(_get_nested_attr(mod, attr_name))
            _del_nested_attr(mod, attr_name)
            _set_nested_attr(mod, attr_name, elem)
    return result


# Adapted from `FunctionalModuleWithBuffers`
# in functorch/functorch/_src/make_functional.py
class FunctionalModuleWithBuffersInInputAndOutput(torch.nn.Module):
    """Given a ``torch.nn.Module``, `create_from` extracts the
    state (params and buffers) and returns a functional version of the model
    ``func`` that can be invoked like a function.

    Compared to `FunctionalModuleWithBuffers` in functorch, the returned
    functional version of the model also has buffers in the output, since
    buffer values can be changed with operations like batchnorm and should be
    tracked as part of output.
    """

    def __init__(self, stateless_model, param_names, buffer_names,
                 param_names_map, buffer_names_map):
        super().__init__()
        self.stateless_model = stateless_model
        self.param_names = param_names
        self.buffer_names = buffer_names

        self.all_names_map = dict(param_names_map)
        self.all_names_map.update(buffer_names_map)

    @staticmethod
    def create_from(model, disable_autograd_tracking=False):
        # TODO: We don't need to copy the model to create a stateless copy
        model_copy = copy.deepcopy(model)
        param_values, param_names, param_names_map = extract_weights(model_copy)
        buffer_values, buffer_names, buffer_names_map = extract_buffers(
            model_copy)
        params = OrderedDict(zip(param_names, param_values))
        buffers = OrderedDict(zip(buffer_names, buffer_values))
        if disable_autograd_tracking:
            for param in param_values:
                param.requires_grad_(False)
        return (
            FunctionalModuleWithBuffersInInputAndOutput(model_copy, param_names,
                                                        buffer_names,
                                                        param_names_map,
                                                        buffer_names_map),
            params,
            buffers,
        )

    def forward(self, params, buffers, *args, **kwargs):
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(self.stateless_model, self.all_names_map,
                                list(params.values()) + list(buffers.values()))
        try:
            return buffers, self.stateless_model(*args, **kwargs)
        finally:
            # Remove the loaded state on self.stateless_model
            _swap_state(self.stateless_model, self.all_names_map, old_state)


def functionalize(module: torch.nn.Module):
    """Returns:
        - `module_func`: a function that has same logic as x.forward but
        callable with either PT or Alpa inputs. It:
            - wraps the original inputs in a tuple
            - takes `params` and `bufs` as extra at beginning of input list
            - produces `bufs` as extra output at beginning of output list
            - all calls are made compatible with Alpa, e.g.:
                - replaces all unexpandable module calls (e.g. nn.Conv2d) with
                  equivalent `torch.*` function calls
                - replaces all torch.nn.functional calls that has in-place ops
                  (e.g. F.batch_norm) with equivalent `atorch.*` function calls
                  that has buffer as part of output
                - complex torch function calls (e.g. F.dropout) are decomposed
                  and implemented with `torch.*` calls
        - `params`: a dict of shape-only tensors representing the trainable
           parameters of the module.
           In PT format if "local", in Alpa format if "dist".
        - `bufs`: a dict of shape-only tensors representing the no-gradient
           parameters of the module.
           In PT format if "local", in Alpa format if "dist".
    Throws error if x.forward:
        - has in-place ops
        - or, has data-dependent control flow
        - or, has other graph-breaking statements (e.g. `print()`) that
          prevents the program from being captured as a single graph
          (only in "dist" mode)
    """

    # This param/buffer name map is used for mapping from FQN in original
    # PyTorch model to FQN in PyTorch FX IR.
    tensor_to_name_map = {}

    all_tensors_pt_orig = dict(named_parameters(module))
    all_tensors_pt_orig.update(dict(named_buffers(module)))

    for k, v in all_tensors_pt_orig.items():
        assert v not in tensor_to_name_map
        tensor_to_name_map[v] = {"orig_name": k}

    def add_transformed_name(tensor_to_name_map, k, v):
        assert v in tensor_to_name_map
        assert "transformed_name" not in tensor_to_name_map[v]
        tensor_to_name_map[v]["transformed_name"] = k

    if atorch.mode() == "dist":
        # In dist mode, use TorchDynamo to enforce:
        # 1) no data-dependent control flow
        # 2) no graph break points
        # 3) no in-place ops

        def convert_pt_module_to_alpa_func(module):
            fx_ir = torch.fx.symbolic_trace(module)

            fx_ir = normalize_ir_no_run(fx_ir)

            # NOTE: due to some unknown reason, only the second normalize pass
            # can convert tensor method to torch function
            # (e.g. `.t()` to `torch.t()`)
            fx_ir = normalize_ir_no_run(fx_ir)

            m_func_name = "_alpa_forward_func"
            m_func_code = fx_ir_to_alpa_func_code(fx_ir, m_func_name)

            if atorch.debug:
                print("JAX function code: ")
                print(m_func_code)

            # pylint: disable=exec-used
            exec(m_func_code)
            module_func = locals()[m_func_name]

            return fx_ir, module_func

        # NOTE: torch.fx.symbolic_trace doesn't hardcode the batch size
        # for `.view()` and `.reshape()` ops, so we DON'T need to trace
        # two graphs (one full-batch, one micro-batch).
        fx_ir, module_func = convert_pt_module_to_alpa_func(module)

        params_pt = dict(named_parameters(fx_ir))
        bufs_pt = dict(named_buffers(fx_ir))

        for k, v in params_pt.items():
            add_transformed_name(tensor_to_name_map, k, v)

        for k, v in bufs_pt.items():
            add_transformed_name(tensor_to_name_map, k, v)

        for k, v in tensor_to_name_map.items():
            if "transformed_name" not in v:
                print(v["orig_name"])

        params_alpa = {
            k: make_shaped_array_from_pt_tensor(v)
            for k, v in params_pt.items()
        }
        bufs_alpa = {
            k: make_shaped_array_from_pt_tensor(v) for k, v in bufs_pt.items()
        }

        if atorch.mode() == "local":
            params = params_pt
            bufs = bufs_pt
        elif atorch.mode() == "dist":
            params = params_alpa
            bufs = bufs_alpa

        name_map = {}
        for elem in tensor_to_name_map.values():
            try:
                name_map[elem["orig_name"]] = elem["transformed_name"]
            except KeyError as e:
                print(f'elem["orig_name"]: {elem["orig_name"]}')
                raise e
    elif atorch.mode() == "local":
        # In local mode, use functionalization pass adapted from functorch
        # TODO: add more rigorous unit tests for this branch
        module_func, params, bufs = \
            FunctionalModuleWithBuffersInInputAndOutput.create_from(module)
        name_map = {}
        for elem in tensor_to_name_map.values():
            name_map[elem["orig_name"]] = elem["orig_name"]

    return module_func, params, bufs, name_map


def meta_init(module_fn: Callable[..., torch.nn.Module], *args, **kwargs):
    pt_module = torchdistx_deferred_init.deferred_init(module_fn, *args,
                                                       **kwargs)
    # pylint: disable=protected-access
    return pt_module._apply(meta_like)

# pylint: disable=line-too-long, unused-argument
"""Maps PyTorch ops to JAX ops"""
import contextlib
import math
from typing import Any, Optional, Sequence, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import torch
from alpa.torch.tensor_utils import numpy_to_torch_dtype_dict


# Adapted from aten/src/ATen/InferSize.h infer_size_impl()
def infer_size(shape, numel):
    newsize = 1
    infer_dim = None
    len(shape)
    res = list(shape)
    for dim in range(len(shape)):
        if shape[dim] == -1:
            if infer_dim is not None:
                raise ValueError("only one dimension can be inferred")
            infer_dim = dim
        elif shape[dim] >= 0:
            newsize *= shape[dim]
        else:
            raise Exception(f"invalid shape dimension {shape[dim]}")

    if (numel == newsize) or (infer_dim is not None and newsize > 0 and
                              numel % newsize == 0):
        if infer_dim is not None:
            # We have a degree of freedom here to select the dimension size;
            # follow NumPy semantics and just bail.  However, a nice error
            # message is needed because users often use `view` as a way to
            # flatten & unflatten dimensions and will otherwise be confused
            # why
            #   empty_tensor.view( 0, 0)
            # works yet
            #   empty_tensor.view(-1, 0)
            # doesn't.
            assert newsize != 0, (
                "cannot reshape tensor of 0 elements into shape " + str(shape) +
                " because the unspecified dimension size -1 can be any " +
                "value and is ambiguous")
            res[infer_dim] = numel // newsize
        return res

    raise Exception(f"shape {shape} is invalid for input of size {numel}")


def init_buffer(
    init_func,
    init_func_kwargs,
    local_rng_seed,
    worker,
    device_id: int,
    shape: Sequence[int],
    dtype: np.dtype,
):

    torch_local_rng = torch.Generator()
    torch_local_rng.manual_seed(local_rng_seed)
    init_func_kwargs["rng"] = torch_local_rng
    init_func_kwargs["shape"] = shape
    init_func_kwargs["dtype"] = numpy_to_torch_dtype_dict[dtype]

    return worker.backend.buffer_from_pyval(init_func(**init_func_kwargs),
                                            worker.local_devices[device_id])


def torch_abs(x):
    return jnp.absolute(x)


def torch_add(x, other):
    return jnp.add(x, other)


def torch_addmm(x, mat1, mat2, beta=1, alpha=1):
    out = alpha * torch.matmul(mat1, mat2)
    if beta == 0:
        return out
    return beta * x + out


def torch_bmm(x, mat2):
    return lax.batch_matmul(x, mat2)


def torch_cat(tensors, dim=0):
    return lax.concatenate(tensors, dim)


def torch_clone(x, memory_format=torch.preserve_format):
    return jnp.array(x, dtype=x.dtype, copy=True, order="K")


def torch_conv2d(x,
                 weight,
                 bias=None,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1):
    # References:
    # - torch-xla impl and haiku / flax impl
    # - https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/convolutions.ipynb
    conv_out = lax.conv_general_dilated(
        x,
        weight,
        stride,
        [(x, x) for x in padding],
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=lax.conv_dimension_numbers(
            x.shape,
            weight.shape,
            ("NCHW", "OIHW",
             "NCHW"),  # TODO: parameterize this! don't assume NCHW format.
        ),
        feature_group_count=groups,
        batch_group_count=1,
    )
    if bias is not None:
        bias_reshaped = bias.reshape(1, bias.shape[0], 1, 1)
        bias_reshaped = jnp.broadcast_to(bias_reshaped, [
            conv_out.shape[0], bias.shape[0], conv_out.shape[2],
            conv_out.shape[3]
        ])
        return conv_out + bias_reshaped
    else:
        return conv_out


def torch_div(x, other, rounding_mode=None):
    ret = None
    if rounding_mode is None:
        ret = jnp.true_divide(x, other)
    elif rounding_mode == "trunc":
        ret = jnp.trunc(jnp.true_divide(x, other))
    elif rounding_mode == "floor":
        ret = jnp.floor_divide(x, other)
    if ret is not None:
        return ret
    else:
        raise NotImplementedError(f"{rounding_mode} is not supported")


def torch_dropout(x, p=0.5, training=True, inplace=False):
    assert not inplace, "Inplace dropout is not supported"
    if p == 0.0:
        return x
    if training:
        # Copied from flax.linen.Dropout impl
        keep_prob = 1.0 - p
        # NOTE: pass None for rng, since Alpa ignores it anyway.
        mask = jax.random.bernoulli(None, p=keep_prob, shape=x.shape)
        return lax.select(mask, x, jnp.zeros_like(x))
    else:
        return x


def torch_exp(x):
    return jnp.exp(x)


def torch_expand(x, sizes):
    computed_sizes = list(sizes)
    for dim, size in enumerate(sizes):
        if size == -1:
            computed_sizes[dim] = x.shape[dim]
    return lax.broadcast_in_dim(x, computed_sizes, list(range(len(x.shape))))


def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool = True):
    if dim_post_expr <= 0:
        assert wrap_scalar
        dim_post_expr = 1
    min_dim = -dim_post_expr
    max_dim = dim_post_expr - 1
    assert not (dim < min_dim or dim > max_dim)
    if dim < 0:
        dim += dim_post_expr
    return dim


def torch_flatten(x, start_dim=0, end_dim=-1):
    input_shape = x.shape
    start_dim = maybe_wrap_dim(start_dim, len(input_shape))
    end_dim = maybe_wrap_dim(end_dim, len(input_shape))
    assert start_dim <= end_dim
    if start_dim == end_dim:
        return x
    slice_numel = 1
    for i in range(start_dim, end_dim + 1):
        slice_numel *= input_shape[i]
    shape = []
    for i in range(start_dim):
        shape.append(input_shape[i])
    shape.append(slice_numel)
    for i in range(end_dim + 1, len(input_shape)):
        shape.append(input_shape[i])
    return torch_view(x, shape)


def torch_full_like(x,
                    fill_value,
                    dtype=None,
                    layout=torch.strided,
                    device=None,
                    requires_grad=False,
                    memory_format=torch.preserve_format):
    return jnp.full_like(x, fill_value, dtype=dtype)


def torch_gelu(x, approximate=False):
    # TODO: use approximate=True or not?
    return jax.nn.gelu(x)


def torch_layer_norm(x,
                     normalized_shape,
                     weight=None,
                     bias=None,
                     eps=1e-05,
                     cudnn_enable=True):
    # TODO: this formula might be wrong
    axis = len(x.shape) - len(normalized_shape)
    mean_val = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.mean((x - mean_val)**2, axis=axis, keepdims=True)
    out = (x - mean_val) / jnp.sqrt(var + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def torch_matmul(x, other):
    return jnp.matmul(x, other)


def torch_max(x, dim=None, keepdim=False):
    return jnp.max(x, axis=dim, keepdims=keepdim)


def torch_mean(x, dim=None, keepdim=False):
    return jnp.mean(x, axis=dim, keepdims=keepdim)


def torch_mm(x, mat2):
    return jnp.matmul(x, mat2)


def torch_mul(x1, x2):
    return jnp.multiply(x1, x2)


def torch_permute(x, dims):
    return jnp.transpose(x, dims)


def torch_pow(x, exponent):
    return jnp.power(x, exponent)


def torch_relu(x):
    return jax.nn.relu(x)


def torch_select(x, dim, index):
    # TODO: likely inefficient. What's the better way?
    return lax.slice_in_dim(x, index, index + 1, stride=1, axis=dim)[0]


def torch_slice(x, dim, start, end, step=1):
    if end > x.shape[dim]:
        end = x.shape[dim]
    return lax.slice_in_dim(x, start, end, stride=step, axis=dim)


def torch_softmax(x, dim):
    x_max = jnp.max(x, axis=dim, keepdims=True)
    unnormalized = jnp.exp(x - x_max)
    return unnormalized / jnp.sum(unnormalized, axis=dim, keepdims=True)


def torch_split(x, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, int):
        split_size = split_size_or_sections
        sections = list(range(split_size, x.shape[dim], split_size))
    else:
        assert isinstance(split_size_or_sections, list)
        sections = split_size_or_sections
    return jnp.split(x, sections, axis=dim)


def torch_sqrt(x):
    return jnp.sqrt(x)


def torch_sub(x, other, alpha=1):
    return x - alpha * other


def torch_sum(x, dim, keepdim=False):
    return jnp.sum(x, axis=dim, keepdims=keepdim)


def torch_t(x):
    return jnp.transpose(x)


def torch_transpose(x, dim0, dim1):
    return jnp.swapaxes(x, dim0, dim1)


def torch_unbind(x, dim=0):
    return tuple(
        jnp.squeeze(p, axis=dim) for p in jnp.split(x, x.shape[dim], axis=dim))


def torch_view(x, shape):
    return lax.reshape(x, infer_size(shape, x.size))


def torch_zeros_like(x,
                     *,
                     dtype=None,
                     layout=None,
                     device=None,
                     requires_grad=False,
                     memory_format=torch.preserve_format):
    return jnp.zeros_like(x, dtype=dtype)


def _normalize(x, mean, var, weight, bias, reduction_axes, feature_axes, eps):
    stats_shape = list(x.shape)
    for axis in reduction_axes:
        stats_shape[axis] = 1
    mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)
    feature_shape = [1] * x.ndim
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
    y = x - mean
    mul = lax.rsqrt(var + eps)
    if weight is not None:
        mul *= weight.reshape(feature_shape)
    y *= mul
    if bias is not None:
        y += bias.reshape(feature_shape)
    return jnp.asarray(y, x.dtype)


def torch_batch_norm(
    x: torch.Tensor,
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    # Ref: https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.BatchNorm.html
    def _abs_sq(x):
        """Computes the elementwise square of the absolute value |x|^2."""
        if jnp.iscomplexobj(x):
            return lax.square(lax.real(x)) + lax.square(lax.imag(x))
        else:
            return lax.square(x)

    def _compute_stats(x,
                       axes,
                       axis_name: Optional[str] = None,
                       axis_index_groups: Any = None):
        # promote x to at least float32, this avoids half precision computation
        # but preserves double or complex floating points
        x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
        mean = jnp.mean(x, axes)
        mean2 = jnp.mean(_abs_sq(x), axes)
        if axis_name is not None:
            concatenated_mean = jnp.concatenate([mean, mean2])
            mean, mean2 = jnp.split(
                lax.pmean(concatenated_mean,
                          axis_name=axis_name,
                          axis_index_groups=axis_index_groups), 2)
        # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
        # to floating point round-off errors.
        var = jnp.maximum(0.0, mean2 - _abs_sq(mean))
        return mean, var

    feature_axes = [1]  # Expect (N, C, ...) shape
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
    feature_shape = [x.shape[ax] for ax in feature_axes]

    if not training:
        mean, var = running_mean, running_var
    else:
        running_mean = jnp.zeros(feature_shape, jnp.float32)
        running_var = jnp.ones(feature_shape, jnp.float32)
        mean, var = _compute_stats(x, reduction_axes)

        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var

    out = _normalize(x, mean, var, weight, bias, reduction_axes, feature_axes,
                     eps)

    return out, running_mean, running_var


def torch_nn_functional_batch_norm(
    x: torch.Tensor,
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    return torch_batch_norm(
        x=x,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )


def torch_nn_functional_dropout(x, p=0.5, training=True, inplace=False):
    return torch_dropout(x, p=p, training=training, inplace=inplace)


def torch_nn_functional_linear(x, weight, bias=None):
    output = torch.matmul(x, torch.t(weight))
    if bias is not None:
        output = output + bias
    return output


def torch_nn_functional_mse_loss(
    x: torch.Tensor,
    target: torch.Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
):
    # TODO: add handling for `size_average` / `reduce` / `reduction`
    return jnp.mean((x - target)**2)


def torch_nn_functional_softmax(x, dim):
    return torch_softmax(x=x, dim=dim)


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = len(tensor.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed "
                         "for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if len(tensor.shape) > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def torch_nn_init_xavier_uniform(x, gain: float = 1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(x)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    useless_key = jax.random.PRNGKey(0)
    return jax.random.uniform(useless_key, x.shape, x.dtype, -a, a)


def torch_nn_init_normal(x, mean: float = 0.0, std: float = 1.0):
    useless_key = jax.random.PRNGKey(0)
    return (jax.random.normal(useless_key, x.shape, x.dtype) + mean) * std


# PyTorch .detach() is equivalent to JAX lax.stop_gradient():
# - https://github.com/google/jax/issues/2025
# PyTorch .view() is equivalent to JAX lax.reshape():
# - https://jax.readthedocs.io/en/latest/_autosummary/lax.reshape.html

op_orig_impl_dict = {}
op_patch_list = [
    (torch, "abs", torch_abs),
    (torch, "add", torch_add),
    (torch, "addmm", torch_addmm),
    (torch, "bmm", torch_bmm),
    (torch, "cat", torch_cat),
    (torch, "clone", torch_clone),
    (torch, "conv2d", torch_conv2d),
    (torch, "div", torch_div),
    (torch, "dropout", torch_dropout),
    (torch, "exp", torch_exp),
    (torch, "expand", torch_expand),
    (torch, "flatten", torch_flatten),
    (torch, "full_like", torch_full_like),
    # (torch, "gelu", torch_gelu),
    (torch, "layer_norm", torch_layer_norm),
    (torch, "matmul", torch_matmul),
    (torch, "max", torch_max),
    (torch, "mean", torch_mean),
    (torch, "mm", torch_mm),
    (torch, "mul", torch_mul),
    (torch, "permute", torch_permute),
    (torch, "pow", torch_pow),
    (torch, "relu", torch_relu),
    (torch, "select", torch_select),
    # (torch, "slice", torch_slice),
    (torch, "softmax", torch_softmax),
    (torch, "split", torch_split),
    (torch, "sqrt", torch_sqrt),
    (torch, "sub", torch_sub),
    (torch, "sum", torch_sum),
    (torch, "t", torch_t),
    (torch, "transpose", torch_transpose),
    (torch, "unbind", torch_unbind),
    (torch, "view", torch_view),
    (torch, "zeros_like", torch_zeros_like),
    (torch.nn.functional, "batch_norm", torch_nn_functional_batch_norm),
    (torch.nn.functional, "dropout", torch_nn_functional_dropout),
    (torch.nn.functional, "linear", torch_nn_functional_linear),
    (torch.nn.functional, "mse_loss", torch_nn_functional_mse_loss),
    (torch.nn.functional, "softmax", torch_nn_functional_softmax),
    (torch.nn.init, "xavier_uniform", torch_nn_init_xavier_uniform),
    (torch.nn.init, "normal", torch_nn_init_normal),
    # TODO: add hard error for in-place ops
]


def patch_ops():
    for python_module, op_name, new_impl in op_patch_list:
        python_module_fqn = str(python_module).split("<module '")[1].split(
            "'")[0]
        op_orig_impl_dict[f"{python_module_fqn}.{op_name}"] = getattr(
            python_module, op_name, None)
        setattr(python_module, op_name, new_impl)


def unpatch_ops():
    for python_module, op_name, _ in op_patch_list:
        python_module_fqn = str(python_module).split("<module '")[1].split(
            "'")[0]
        op_orig_impl = op_orig_impl_dict.get(f"{python_module_fqn}.{op_name}",
                                             None)
        if op_orig_impl is not None:
            setattr(python_module, op_name, op_orig_impl)
        else:
            delattr(python_module, op_name)


@contextlib.contextmanager
def bind_ops(enabled=True):
    """Context manager within which many PyTorch ops are monkey-patched
    to support distributed computation with Alpa.
    """
    if enabled:
        patch_ops()
    try:
        yield
    finally:
        if enabled:
            unpatch_ops()


def enable_dist_for_func(func: Callable = None):
    """Returns a callable that executes `func` within `bind_ops` context.
    """

    def wrapped_func(*args, **kwargs):
        with bind_ops():
            return func(*args, **kwargs)

    return wrapped_func

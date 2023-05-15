# flake8: noqa
from collections import OrderedDict
from dataclasses import fields
import functools
from typing import Any, Callable, Optional, Tuple, Optional, Union, Sequence

from alpa.api import value_and_grad
import flax
from flax.training import train_state, dynamic_scale as dynamic_scale_lib
from flax.training.dynamic_scale import DynamicScaleResult
from flax import struct
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import jaxlib.xla_extension as jax_xla
import optax

Array = Any


def is_tensor(x):
    """
    Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor`, obj:`jax.Tensor` or
    :obj:`np.ndarray`.
    """
    #if is_torch_fx_proxy(x):
    #    return True
    #if is_torch_available():
    #    import torch

    #    if isinstance(x, torch.Tensor):
    #        return True
    #if is_tf_available():
    #    import tensorflow as tf

    #    if isinstance(x, tf.Tensor):
    #        return True

    #if is_flax_available():
    if True:
        import jaxlib.xla_extension as jax_xla
        from jax.core import Tracer

        if isinstance(x, (jax.Array, Tracer)):
            return True

    return isinstance(x, np.ndarray)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a regular
    python dictionary.
    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


@flax.struct.dataclass
class FlaxBaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args:
        last_hidden_state (:obj:`jax.Array` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(jax.Array)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jax.Array` (one for the output of the embeddings + one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(jax.Array)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jax.Array` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: jax.Array = None
    hidden_states: Optional[Tuple[jax.Array]] = None
    attentions: Optional[Tuple[jax.Array]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    Args:
        last_hidden_state (:obj:`jax.Array` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`jax.Array` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.
        hidden_states (:obj:`tuple(jax.Array)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jax.Array` (one for the output of the embeddings + one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(jax.Array)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jax.Array` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: jax.Array = None
    pooler_output: jax.Array = None
    hidden_states: Optional[Tuple[jax.Array]] = None
    attentions: Optional[Tuple[jax.Array]] = None


@flax.struct.dataclass
class FlaxBertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.BertForPreTraining`.
    Args:
        prediction_logits (:obj:`jax.Array` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`jax.Array` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (:obj:`tuple(jax.Array)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jax.Array` (one for the output of the embeddings + one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(jax.Array)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jax.Array` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    prediction_logits: jax.Array = None
    seq_relationship_logits: jax.Array = None
    hidden_states: Optional[Tuple[jax.Array]] = None
    attentions: Optional[Tuple[jax.Array]] = None


@flax.struct.dataclass
class FlaxMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.
    Args:
        logits (:obj:`jax.Array` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(jax.Array)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jax.Array` (one for the output of the embeddings + one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(jax.Array)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jax.Array` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: jax.Array = None
    hidden_states: Optional[Tuple[jax.Array]] = None
    attentions: Optional[Tuple[jax.Array]] = None


@flax.struct.dataclass
class FlaxSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        logits (`jnp.ndarray` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


def softmax_cross_entropy(logits, labels):
    return -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)


class TrainState(train_state.TrainState):
    """This is an extended version of flax.training.train_state.TrainState.

    This class wraps the logic for creating the master weight copy in
    mixed precision training.
    """
    master_copy: flax.core.FrozenDict[str, Any]
    dynamic_scale: Optional[dynamic_scale_lib.DynamicScale]

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.
        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.
        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.
        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """
        if self.master_copy is None:
            master_params = self.params
        else:
            master_params = self.master_copy

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                master_params)
        new_master_params = optax.apply_updates(master_params, updates)

        if self.master_copy is None:
            new_master_copy = None
            new_params = new_master_params
        else:
            new_master_copy = new_master_params
            new_params = jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=jnp.float16), new_master_params)

            # A hack to make the donation works perfectly in gradient accumulation:
            # We need the accumulate_grad to take the old params as input.
            new_params_flat, tree = jax.tree_util.tree_flatten(new_params)
            old_params_flat, _ = jax.tree_util.tree_flatten(self.params)
            new_params_flat = [
                x + 0.0 * y for x, y in zip(new_params_flat, old_params_flat)
            ]
            new_params = jax.tree_util.tree_unflatten(tree, new_params_flat)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            master_copy=new_master_copy,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, use_master_copy=False, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        if use_master_copy:
            master_copy = jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=jnp.float32), params)
            params = jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=jnp.float16), params)
            opt_state = tx.init(master_copy)
        else:
            master_copy = None
            opt_state = tx.init(params)

        return cls(
            step=np.array(0, dtype=np.int32),
            apply_fn=apply_fn,
            params=params,
            master_copy=master_copy,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    @classmethod
    def create_aval(cls,
                    *,
                    apply_fn,
                    params,
                    tx,
                    use_master_copy=False,
                    **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = jax.eval_shape(tx.init, params)

        if use_master_copy:
            master_copy = params
            params = jax.eval_shape(
                lambda p: jax.tree_util.tree_map(
                    lambda x: jnp.asarray(x, dtype=jnp.float16), p), params)
        else:
            master_copy = None

        return cls(
            step=np.array(0, dtype=np.int32),
            apply_fn=apply_fn,
            params=params,
            master_copy=master_copy,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


class DynamicScale(struct.PyTreeNode):
    """This is the same as flax.optim.DynamicScale, except that
  jax.value_and_grad is replaced by alpa.value_and_grad.

  Dynamic loss scaling for mixed precision gradients.

  For many models gradient computations in float16 will result in numerical
  issues because small/large gradients being flushed to zero/infinity.
  Dynamic loss scaling is an algorithm that aims to find the largest scalar
  multiple for which the gradient does not overflow. This way the risk of
  underflow is minimized.

  the `value_and_grad` method mimicks `jax.value_and_grad`. Beside the loss
  and gradients it also ouputs and updated `DynamicScale` instance with the
  current loss scale factor. This method also returns a boolean value indicating
  whether the gradients are finite.

  Example::

    def loss_fn(p):
      return jnp.asarray(p, jnp.float16) ** 2
    p = jnp.array(1., jnp.float32)

    dyn_scale = optim.DynamicScale(growth_interval=10)
    compute_grad = jax.jit(lambda ds, p: ds.value_and_grad(loss_fn)(p))
    for _ in range(100):
      dyn_scale, is_fin, loss, grad = compute_grad(dyn_scale, p)
      p += jnp.where(is_fin, 0.01 * grad, 0.)
      print(loss)

  Jax currently cannot execute conditionals efficiently on GPUs therefore we
  selectifly ignore the gradient update using `jax.numpy.where` in case of
  non-finite gradients.

  Attributes:
    growth_factor: how much to grow the scalar after a period of finite
      gradients (default: 2.).
    backoff_factor: how much to shrink the scalar after a non-finite gradient
      (default: 0.5).
    growth_interval: after how many steps of finite gradients the scale should
      be increased (default: 2000).
    fin_steps: indicates how many gradient steps in a row have been finite.
    scale: the current scale by which the loss is multiplied.
  """
    growth_factor: float = struct.field(pytree_node=False, default=2.0)
    backoff_factor: float = struct.field(pytree_node=False, default=0.5)
    growth_interval: int = struct.field(pytree_node=False, default=2000)
    fin_steps: Array = 0
    scale: Array = 65536.0

    def value_and_grad(
        self,
        fun: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
    ) -> Callable[..., DynamicScaleResult]:
        """Wrapper around `jax.value_and_grad`.

    Args:
      fun: Function to be differentiated. Its arguments at positions specified
        by ``argnums`` should be arrays, scalars, or standard Python containers.
        It should return a scalar (which includes arrays with shape ``()``
        but not arrays with shape ``(1,)`` etc.)
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default 0).
      has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where
        the first element is considered the output of the mathematical function
        to be differentiated and the second element is auxiliary data.
        Default False.
      axis_name: If an axis is given the gradients will be averaged across
        replicas (default: None).
    Returns:
      A function that takes the same arguments as `fun` and
      returns a DynamicScaleResult
    """

        @functools.wraps(fun)
        def loss_wrapper(*args):
            aux = fun(*args)
            if has_aux:
                return (self.scale * aux[0], aux[1])
            else:
                return self.scale * aux

        grad_fn = value_and_grad(loss_wrapper, argnums, has_aux)

        def grad_fn_wrapper(*args):
            aux, grad = grad_fn(*args)
            aux = (aux[0] / self.scale, aux[1]) if has_aux else aux / self.scale

            grad = jax.tree_util.tree_map(
                lambda g: jnp.asarray(g, jnp.float32) / self.scale, grad)
            if axis_name is not None:
                grad = lax.pmean(grad, axis_name)

            finite = jnp.array(True)
            for g in jax.tree_util.tree_leaves(grad):
                finite &= jnp.all(lax.is_finite(g))

            grow = self.fin_steps == self.growth_interval
            fin_scale = jnp.where(grow & finite,
                                  self.scale * self.growth_factor, self.scale)
            inf_scale = self.scale * self.backoff_factor
            new_scale = jnp.where(finite, fin_scale, inf_scale)
            new_fin_steps = jnp.where(grow | (~finite), 0, self.fin_steps + 1)

            new_self = self.replace(fin_steps=new_fin_steps, scale=new_scale)
            return DynamicScaleResult(new_self, finite, aux, grad)

        return grad_fn_wrapper

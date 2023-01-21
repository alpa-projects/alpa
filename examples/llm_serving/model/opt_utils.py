from functools import partial

import jax
from jax import xla, jit
from jax.core import Primitive
from jax._src.lib import xla_client as xc
from dataclasses import dataclass


def sync(device_id=0):
    jax.devices()[device_id].synchronize_all_activity()
    return


@dataclass
class TransformerModelConfig:
    # hidden size
    H: int = 768
    # number of layers
    L: int = 12
    # number of attention heads
    n_head: int = 12
    seq_len: int = 2048
    vocab_size: int = 50272


def compute_gpt_tflops_inference_with_padding(batch_size, gen_len, seq_len,
                                              num_layers, hidden_size,
                                              vocab_size, num_gpus, latency):
    """This calculation assumes that each code decoded attend to seq_len number tokens."""
    factor = 24
    total_flop = factor * batch_size * gen_len * (hidden_size ** 2) * num_layers * \
          (1 + seq_len / (6 * hidden_size)) \
          + 2 * batch_size * gen_len * hidden_size * vocab_size
    # Note (Hao): it should be 4 here because of input embedding, but we will
    # respect Deepak's eq. instead.
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops


def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)


index_select_p = Primitive("index-select")


@partial(jit, static_argnums=(2,))
def jax_index_select(input, index, dim=0):
    return index_select_p.bind(input, index, dim=dim)


def _index_select_eval(input, index, dim):
    return input


def _index_select_translation(c, input, index, dim):
    return xc.ops.IndexSelect(input, index, dim)


index_select_p.def_abstract_eval(_index_select_eval)
index_select_p.def_impl(partial(xla.apply_primitive, index_select_p))
xla.translations[index_select_p] = _index_select_translation

from functools import partial

import jax
import jax.numpy as jnp

from EasyLM.models.llama.llama_model import FlaxLLaMAForCausalLMModule


def do_monkey_patch():
    # TODO: jax 0.3.22 does not support eval shape with static args well. Remove
    # after rebasing to jax 0.4, use the model's _do_init=False then.
    def init_dummy(self, *args, **kwargs):
        avals = jax.eval_shape(partial(self._backup_init, **kwargs), *args)
        return jax.tree_util.tree_map(lambda x: jnp.full(x.shape, 1e-8, x.dtype),
                                    avals)
    if not hasattr(FlaxLLaMAForCausalLMModule, "_backup_init"):
        FlaxLLaMAForCausalLMModule._backup_init = FlaxLLaMAForCausalLMModule.init
    FlaxLLaMAForCausalLMModule.init = init_dummy
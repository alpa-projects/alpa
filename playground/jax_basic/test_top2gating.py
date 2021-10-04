import jax
import jax.numpy as jnp
from jax.nn import one_hot

key = jax.random.PRNGKey(0)

G = 4
S = 8
E = 2
C = 2 * S // E

gates = jax.random.uniform(key, (G, S, E))

def top2_gating(gates):  # (G, S, E)
    index_1 = jnp.argmax(gates, axis=-1)  # (G, S)
    mask_1 = one_hot(index_1, E)          # (G, S, E)
    gate_1 = jnp.einsum("GSE,GSE->GS", gates, mask_1)  # (G, S)

    gates_without_top_1 = gates * (1 - mask_1)

    index_2 = jnp.argmax(gates_without_top_1, axis=-1)  # (G, S, E)
    mask_2 = one_hot(index_2, E)
    gate_2 = jnp.einsum("GSE,GSE->GS", gates_without_top_1, mask_2)

    pos_1 = jnp.cumsum(mask_1, axis=-2) - mask_1
    mask_1 *= pos_1 < C
    pos_1 = jnp.einsum("GSE,GSE->GS", pos_1, mask_1)

    mask_1_count = jnp.sum(mask_1, axis=-2)
    mask_1_flat = jnp.sum(mask_1, axis=-1)

    pos_2 = (jnp.cumsum(mask_2, axis=-2) - mask_2) + jnp.expand_dims(mask_1_count, -2)
    mask_2 *= pos_2 < C
    pos_2 = jnp.einsum("GSE,GSE->GS", pos_2, mask_2)

    mask_2_flat = jnp.sum(mask_2, axis=-1)

    gate_1 *= mask_1_flat
    gate_2 *= mask_2_flat

    denom = gate_1 + gate_2
    denom = jnp.where(denom > 0, denom, jnp.ones_like(denom))
    gate_1 /= denom
    gate_2 /= denom

    b = one_hot(pos_1, E)
    a = jnp.expand_dims(gate_1 * mask_1_flat, -1) * one_hot(index_1, E)
    first_part_of_combine_tensor = jnp.einsum("GSE,GSC->GSEC", a, b)

    b = one_hot(pos_2, E)
    a = jnp.expand_dims(gate_2 * mask_2_flat, -1) * one_hot(index_2, E)
    second_part_of_combine_tensor = jnp.einsum("GSE,GSC->GSEC", a, b)

    combined_tensor = first_part_of_combine_tensor + second_part_of_combine_tensor
    dispatch_tensor = combined_tensor.astype(jnp.bool_)

    return combined_tensor, dispatch_tensor

combined_tensor, dispatch_tensor = top2_gating(gates)
print(combined_tensor.shape)


import numpy as np
import jax.numpy as jnp

data=[4,1024,4096*3]

qvk_combined_states=np.zeros(data)
qvk_combined_states = qvk_combined_states.reshape(
    qvk_combined_states.shape[:2] + (-1, 3))
print(qvk_combined_states.shape)


query_states, value_states, key_states = jnp.split(qvk_combined_states,
                                                3,
                                                axis=3)

print(query_states.shape)

query_states = query_states.reshape((4,1024,4096)[:2] +
                                    (32,128))
print(query_states.shape)
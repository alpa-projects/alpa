import jax
import cloudpickle as pickle
@jax.jit
def f(x):
    return x * 13
pickle.dumps(f)
print('OK')

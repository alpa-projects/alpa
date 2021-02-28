import numpy as np
import jax
from jax.lib import xla_client as xc


def test_sin_cos():
    def f(x):
        return jax.numpy.sin(jax.numpy.cos(x.T))

    c = jax.xla_computation(f)(np.ones((10,8)))

    print(c.as_hlo_text())

    gpu_backend = xc.get_local_backend("gpu")
    compiled_computation = gpu_backend.compile(c)

    host_input = np.ones((10,8), dtype=np.float32)
    device_input = gpu_backend.buffer_from_pyval(host_input)
    device_out = compiled_computation.execute([device_input ,])

    print(device_out[0].to_py())


if __name__ == "__main__":
    #test_sin_cos()

    test_parallel()


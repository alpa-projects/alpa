import time

import jax
import alpa
from alpa.util import compute_bytes

from opt_model import get_config, init_model_aval, load_np_params


GB = 1 << 30


def test_save_np_to_ts(name):
    config = get_config(name)
    np_weights_folder = f"/home/ubuntu/opt_weights/{name}_np"
    ts_weights_folder = f"/home/ubuntu/opt_weights/{name}_ts"

    model, params = init_model_aval(config)

    # Load model
    print("Load model...")
    tic = time.time()
    params = load_np_params(params, np_weights_folder, config)
    num_bytes = compute_bytes(params)
    duration = time.time() - tic
    print(f"Duration: {duration:.2f}, Bandwidth: {num_bytes / duration / GB:.2f} GB/s")

    # Save model
    print("Save model...")
    tic = time.time()
    alpa.save_checkpoint(ts_weights_folder, params, 1)
    duration = time.time() - tic
    print(f"Duration: {duration:.2f}, Bandwidth: {num_bytes / duration / GB:.2f} GB/s")



if __name__ == "__main__":
    name = "175B"
    test_save_np_to_ts(name)

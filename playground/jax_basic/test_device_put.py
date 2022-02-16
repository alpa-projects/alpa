import time

import jax
import jax.numpy as jnp

import torch
import numpy as np

def benchmark_func(func):
    warmup = 1
    number = 2

    for i in range(warmup):
        func()
    jax.local_devices()[0].synchronize_all_activity()

    tic = time.time()
    for i in range(number):
        func()
    toc = time.time()

    return (toc - tic) / number



if __name__ == "__main__":
    num_samples = 20000
    batch_size = 2048

    print("Init data...")
    np.random.seed(0)
    images = np.ones((num_samples, 224, 224, 3), dtype=np.float32)
    labels = np.ones((num_samples,), dtype=np.int32)
    steps_per_epoch = len(images) // batch_size

    devices = jax.devices()

    print("Load data...")
    shard_size = batch_size // len(devices)

    def np_array_view():
        for i in range(steps_per_epoch):
            batch_images = images[i * batch_size: (i+1)*batch_size]
            batch_labels = labels[i * batch_size: (i+1)*batch_size]

    def np_array_copy():
        for i in range(steps_per_epoch):
            batch_images = np.array(images[i * batch_size: (i+1)*batch_size])
            batch_labels = np.array(labels[i * batch_size: (i+1)*batch_size])

    def jnp_array_copy():
        for i in range(steps_per_epoch):
            batch_images = images[i * batch_size: (i+1)*batch_size]
            batch_labels = labels[i * batch_size: (i+1)*batch_size]
            batch_images = jnp.array(batch_images)
            batch_labels = jnp.array(batch_labels)

    signal = jnp.ones((1024, 1024))

    def jax_device_put():
        for i in range(steps_per_epoch):
            batch_images = images[i * batch_size: (i+1)*batch_size]
            batch_labels = labels[i * batch_size: (i+1)*batch_size]
            jax.device_put(batch_images)
            jax.device_put(batch_labels)
            signal.block_until_ready()

    def jax_device_put2():
        for i in range(steps_per_epoch):
            batch_images = images[i * batch_size: (i+1)*batch_size]
            batch_labels = labels[i * batch_size: (i+1)*batch_size]
            jax.device_put(batch_images)
            jax.device_put(batch_labels)
            signal.block_until_ready()

    def jax_device_put_sync():
        for i in range(steps_per_epoch):
            batch_images = images[i * batch_size: (i+1)*batch_size]
            batch_labels = labels[i * batch_size: (i+1)*batch_size]
            x = jax.device_put(batch_images)
            jax.device_put(batch_labels)
            x.block_until_ready()

    def jax_device_put_multi_devices():
        for i in range(steps_per_epoch):
            batch_images = images[i * batch_size: (i+1)*batch_size]
            batch_labels = labels[i * batch_size: (i+1)*batch_size]
            for j, d in enumerate(devices):
                jax.device_put(batch_images[j * shard_size:(j+1) * shard_size], d)
                jax.device_put(batch_labels[j * shard_size:(j+1) * shard_size], d)

    def jax_device_put_multi_devices_slow():
        for i in range(steps_per_epoch):
            batch_images = images[i * batch_size: (i+1)*batch_size]
            batch_labels = labels[i * batch_size: (i+1)*batch_size]
            for j, d in enumerate(devices):
                jax.device_put(batch_images[j * shard_size:(j+1) * shard_size], d)
                jax.device_put(batch_labels[j * shard_size:(j+1) * shard_size], d)

    def jax_device_put_multi_devices_sync():
        arrays = [None] * len(devices)
        for i in range(steps_per_epoch):
            batch_images = images[i * batch_size: (i+1)*batch_size]
            batch_labels = labels[i * batch_size: (i+1)*batch_size]

            for j, d in enumerate(devices):
                arrays[j] = jax.device_put(batch_images[j * shard_size:(j+1) * shard_size], d)
                jax.device_put(batch_labels[j * shard_size:(j+1) * shard_size], d)

            for j in range(len(devices)):
                arrays[j].block_until_ready()

    def jax_device_put_multi_devices_sync_serial():
        arrays = [None] * len(devices)
        for i in range(steps_per_epoch):
            batch_images = images[i * batch_size: (i+1)*batch_size]
            batch_labels = labels[i * batch_size: (i+1)*batch_size]

            for j, d in enumerate(devices):
                arrays[j] = jax.device_put(batch_images[j * shard_size:(j+1) * shard_size], d)
                jax.device_put(batch_labels[j * shard_size:(j+1) * shard_size], d)
                arrays[j].block_until_ready()

    #time_np_array_view = benchmark_func(np_array_view)
    #time_np_array_copy = benchmark_func(np_array_copy)
    #time_jnp_array_copy = benchmark_func(jnp_array_copy)
    time_jax_device_put = benchmark_func(jax_device_put)
    time_jax_device_put2 = benchmark_func(jax_device_put2)
    time_jax_device_put_sync = benchmark_func(jax_device_put_sync)
    time_jax_device_put_multi_devices = benchmark_func(jax_device_put_multi_devices)
    time_jax_device_put_multi_devices_slow = benchmark_func(jax_device_put_multi_devices_slow)
    time_jax_device_put_multi_devices_sync = benchmark_func(jax_device_put_multi_devices_sync)
    time_jax_device_put_multi_devices_sync_serial = benchmark_func(jax_device_put_multi_devices_sync_serial)

    print(f"Steps: {steps_per_epoch}")
    #print(f"np_array_view: {time_np_array_view * 1e3:.3f} ms")
    #print(f"np_array_copy: {time_np_array_copy * 1e3:.3f} ms")
    #print(f"jnp_array_copy: {time_jnp_array_copy * 1e3:.3f} ms")
    print(f"jax_device_put: {time_jax_device_put * 1e3:.3f} ms")
    print(f"jax_device_put2: {time_jax_device_put2 * 1e3:.3f} ms")
    print(f"jax_device_put_sync: {time_jax_device_put_sync * 1e3:.3f} ms")
    print(f"jax_device_put_multi_devices: {time_jax_device_put_multi_devices* 1e3:.3f} ms")
    print(f"jax_device_put_multi_devices_slow: {time_jax_device_put_multi_devices_slow * 1e3:.3f} ms")
    print(f"jax_device_put_multi_devices_sync: {time_jax_device_put_multi_devices_sync * 1e3:.3f} ms")
    print(f"jax_device_put_multi_devices_sync_serial: {time_jax_device_put_multi_devices_sync_serial * 1e3:.3f} ms")


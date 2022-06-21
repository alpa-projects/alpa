import os
import subprocess
import time

from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

import alpa
from alpa import save_checkpoint as alpa_save_checkpoint
from alpa import restore_checkpoint as alpa_restore_checkpoint
from alpa import PipeshardParallel, DistributedArray
from alpa.testing import (MLPModel, create_train_state, get_mlp_train_step)
from alpa.device_mesh import get_global_cluster


def _get_efs_mount_point():
    # Hacky function to get the EFS mount point
    for line in subprocess.check_output("df -h",
                                        shell=True).decode().split('\n'):
        cols = line.split(' ')
        if "efs" in cols[0]:
            return cols[-1] + "/"
    return None


def _get_save_prefix(to_efs):
    if to_efs:
        # Get EFS mount point for the multi-host test
        save_prefix = _get_efs_mount_point()
    else:
        # Use tmp dir for the single-host test
        save_prefix = "/tmp/"
    return save_prefix


LOOP_CNT = 2


def benchmark_ndarray_save_load(mode="flax", to_efs=True):
    """
    EFS performance: https://docs.aws.amazon.com/efs/latest/ug/performance.html

    if mode == "flax": use flax.training.checkpoints.save_checkpoint/restore_checkpoint
    elif mode == "alpa": use alpa.serialization.save_checkpoint/restore_checkpoint
    elif mode == "numpy: use np.save/load

    Benchmark results on EFS: 
    - flax.save_checkpoint:    save average run time: 15.0580 seconds, save average throughput: 0.5313 Gbps
    - flax.restore_checkpoint: load average run time:  6.8287 seconds, load average throughput: 1.2225 Gbps

    - alpa.save_checkpoint:    save average run time: 12.8583 seconds, save average throughput: 0.6222 Gbps
                 use cache:    
    - alpa.restore_checkpoint: N/A

    - np.save:                 save average run time: 10.4157 seconds, save average throughput: 0.7682 Gbps
    - np.load:                 load average run time:  2.9987 seconds, load average throughput: 4.9950 Gbps

    Benchmark results on local filesystem:
    - flax.save_checkpoint:    save average run time: 5.5268 seconds, save average throughput: 1.4475 Gbps
    - flax.restore_checkpoint: load average run time: 5.1856 seconds, load average throughput: 1.5428 Gbps

    - alpa.save_checkpoint:    save average run time: 10.3145 seconds, save average throughput: 0.7756 Gbps
    - alpa.restore_checkpoint: N/A

    - np.save:                 save average run time: 0.8104 seconds, save average throughput:  9.8718 Gbps
    - np.load:                 load average run time: 0.7327 seconds, load average throughput: 10.9179 Gbps
    """
    rngkey = random.PRNGKey(0)
    #arr_sizes = [1024*1024, 4*1024*1024, 16*1024*1024, 32*1024*1024] # 4M, 16M, 64M, 128M
    arr_sizes = [256 * 1024 * 1024]  # 1G
    benchmark_arrs = [
        random.normal(rngkey, (arr_size,)) for arr_size in arr_sizes
    ]
    for arr in benchmark_arrs:
        save_tot_duration = 0.0
        save_tot_throughput = 0.0
        load_tot_duration = 0.0
        load_tot_throughput = 0.0
        prefix = _get_save_prefix(to_efs)
        for i in range(LOOP_CNT):
            assert (prefix is not None)
            outdir = os.path.join(prefix, "benchmark_checkpoint")
            # clean working directory
            subprocess.run(["rm", "-rf", outdir])
            # rebuild working directory
            os.mkdir(outdir)
            print(f"save to {outdir}")
            ckpt_path = os.path.join(outdir, "checkpoint_1.npy")  # numpy-only

            # save benchmark
            start = time.time()
            if mode == "flax":
                save_checkpoint(outdir, arr, i)
            elif mode == "alpa":
                alpa_save_checkpoint(outdir, arr, i, "/tmp")
            else:
                np.save(ckpt_path, arr)
            duration = time.time() - start
            throughput = arr.size * 32 / 1024 / 1024 / 1024 / duration
            if i >= 1:
                save_tot_duration += duration
                save_tot_throughput += throughput
            print(
                f"loop {i} save, time: {duration:.4f} seconds, throughput: {throughput:.4f} Gbps"
            )

            gpus = jax.devices("gpu")
            # load benchmark
            start = time.time()
            if mode == "flax":
                restore_checkpoint(outdir, None, None)
            elif mode == "alpa":
                print("alpa skip load array benchmark")
                continue
            else:
                jax.block_until_ready(
                    jax.device_put(np.load(ckpt_path), gpus[0]))

            duration = time.time() - start
            throughput = arr.size * 32 / 1024 / 1024 / 1024 / duration
            if i >= 1:
                load_tot_duration += duration
                load_tot_throughput += throughput
            print(
                f"loop {i} load, time: {duration:.4f} seconds, throughput: {throughput:.4f} Gbps"
            )

        print(
            f"save average run time: {save_tot_duration/(LOOP_CNT - 1):.4f} seconds, save average throughput: {save_tot_throughput/(LOOP_CNT - 1):.4f} Gbps"
        )
        print(
            f"load average run time: {load_tot_duration/(LOOP_CNT - 1):.4f} seconds, load average throughput: {load_tot_throughput/(LOOP_CNT - 1):.4f} Gbps"
        )


def count_params(model):
    return sum(x.size for x in jax.tree_leaves(model))


def benchmark_mlp_save(mode="flax", to_efs=True):
    """
    Benchmark results on EFS: 
    - flax.save_checkpoint: average run time: 45.19087886810303 seconds, average throughput: 0.5313484040513637 Gbps
    - alpa.save_checkpoint: average run time: 16.15189399719238, average throughput: 1.4860819837013484 Gbps
                 use cache: 
    - np.save:              average run time: 20.618193340301513, average throughput: 1.1642373201358331 Gbps

    Benchmark results on local disk:
    - flax.save_checkpoint: average run time: 16.1341721534729, average throughput: 1.4877078603042466 Gbps
    - alpa.save_checkpoint: average run time: 10.663438653945922, average throughput: 2.2509621962263244 Gbps
    - np.save:              average run time: 20.618193340301513, average throughput: 1.1642373201358331 Gbps
    """
    # Init model and optimizer
    batch_size = 64
    hidden_dim = 8192  # 3072M
    input_dim = output_dim = hidden_dim
    model = MLPModel(hidden_dim=hidden_dim,
                     output_dim=output_dim,
                     manual_pipeline_layer=True)

    # Init batch args
    rngkey = random.PRNGKey(0)
    x = random.normal(rngkey, (batch_size, input_dim), jnp.float32)
    state = create_train_state(rngkey, model, [x])
    model_size = count_params(state)
    print(f"model size: {model_size * 4 / 1024 / 1024} MB")

    tot_duration = 0.0
    tot_throughput = 0.0
    prefix = _get_save_prefix(to_efs)
    for i in range(LOOP_CNT):
        assert (prefix is not None)
        outdir = os.path.join(prefix, "benchmark_checkpoint")
        ckpt_path = os.path.join(outdir, f"checkpoint_1.npy")  # numpy-only
        # clean working directory
        subprocess.run(["rm", "-rf", outdir])
        # rebuild working directory
        os.mkdir(outdir)
        print(f"save to {outdir}")

        start = time.time()
        if mode == "flax":
            save_checkpoint(outdir, state, i)
        elif mode == "alpa":
            alpa_save_checkpoint(outdir, state, i, "/tmp")
        else:
            np.save(ckpt_path, state.params)
            np.save(ckpt_path, state.opt_state)
        duration = time.time() - start

        throughput = model_size * 32 / 1024 / 1024 / 1024 / duration
        tot_duration += duration
        tot_throughput += throughput
        print(
            f"loop {i}, time: {duration} seconds, throughput: {throughput} Gbps"
        )
    print(
        f"average run time: {tot_duration/LOOP_CNT}, average throughput: {tot_throughput/LOOP_CNT} Gbps"
    )


def benchmark_dist_arr_save(to_efs=False):
    """
    Benchmark results on local disk:
    - one host:
        - TensorStore: save average run time: 9.9292 seconds, save average throughput: 0.8057 Gbps
        - np.save      save average run time: 0.8113 seconds, save average throughput: 9.8601 Gbps

    - two hosts:
        - TensorStore: save average run time: 3.9092 seconds, save average throughput: 2.0465 Gbps
        - np.save:     save average run time: 0.4702 seconds, save average throughput: 17.0149 Gbps
    """
    device_cluster = get_global_cluster()
    physical_mesh = device_cluster.get_physical_mesh()
    logical_mesh = physical_mesh.get_logical_mesh()

    rngkey = random.PRNGKey(0)
    arr_shape = (64 * 1024, 16 * 1024)  #1GB
    arr = random.normal(rngkey, arr_shape)

    sharding_spec = logical_mesh.make_tile_spec(arr, [0, 1], [0, 1])
    input_indices = sharding_spec.indices(arr.shape).flatten()
    (dist_arr,) = physical_mesh.shard_args_to_arrays(
        (jax.ShapedArray(arr.shape, jnp.int32),), (input_indices,),
        (sharding_spec,), (arr,))

    save_tot_duration = 0.0
    save_tot_throughput = 0.0
    outdir = "/tmp/benchmark_save"
    for i in range(LOOP_CNT):
        # Save the DistributedArray (one replica only)
        subprocess.run(["rm", "-rf", outdir])
        print(f"save to {outdir}")

        start = time.time()
        jax.block_until_ready(dist_arr.save(outdir))
        duration = time.time() - start
        throughput = arr.size * 32 / 1024 / 1024 / 1024 / duration
        if i >= 1:
            save_tot_duration += duration
            save_tot_throughput += throughput
        print(
            f"loop {i} save, time: {duration:.4f} seconds, throughput: {throughput:.4f} Gbps"
        )
    print(
        f"save average run time: {save_tot_duration/(LOOP_CNT - 1):.4f} seconds, save average throughput: {save_tot_throughput/(LOOP_CNT - 1):.4f} Gbps"
    )


def benchmark_dist_arr_load():
    """
    Benchmark results on local disk:
    - one host:
        TensorStore: load average run time: 4.0709 seconds, load average throughput: 1.9651 Gbps
        np.load:     load average run time: 1.5235 seconds, load average throughput: 5.2512 Gbps
    
    - two hosts:
        TensorStore: load average run time: 3.6650 seconds, load average throughput: 2.1828 Gbps
        np.load:     load average run time: 0.7644 seconds, load average throughput: 10.4655 Gbps
    """
    device_cluster = get_global_cluster()
    physical_mesh = device_cluster.get_physical_mesh()
    logical_mesh = physical_mesh.get_logical_mesh()

    rngkey = random.PRNGKey(0)
    arr_shape = (64 * 1024, 16 * 1024)  #1GB
    arr = random.normal(rngkey, arr_shape)

    sharding_spec = logical_mesh.make_tile_spec(arr, [0, 1], [0, 1])

    load_tot_duration = 0.0
    load_tot_throughput = 0.0
    outdir = "/tmp/benchmark_save"
    for i in range(LOOP_CNT):
        print(f"load from {outdir}")

        # load benchmark
        start = time.time()
        print("start", time.time())
        jax.block_until_ready(
            DistributedArray.load(outdir, jax.ShapedArray(arr.shape, jnp.int32),
                                  physical_mesh, sharding_spec))
        print("end", time.time())
        duration = time.time() - start
        throughput = arr.size * 32 / 1024 / 1024 / 1024 / duration
        if i >= 1:
            load_tot_duration += duration
            load_tot_throughput += throughput
        print(
            f"loop {i} load, time: {duration:.4f} seconds, throughput: {throughput:.4f} Gbps"
        )
    print(
        f"load average run time: {load_tot_duration/(LOOP_CNT - 1):.4f} seconds, load average throughput: {load_tot_throughput/(LOOP_CNT - 1):.4f} Gbps"
    )


def benchmark_mlp_dist_save():
    """
    Benchmark results on EFS:
    - alpa.save_checkpoint:
        save average run time: 161.8653 seconds, save average throughput: 0.1483 Gbps
        load average run time:  40.2772 seconds, load average throughput: 0.5965 Gbps
    
    Benchmark results on local disk:
    - one host:
        np.save (batch version) save average run time: 1.3313 seconds, save average throughput: 18.0300 Gbps

    - two hosts:
        TensorStore:            save average run time: 19.9880 seconds, save average throughput: 1.2009 Gbps
        np.save:                save average run time:  2.4631 seconds, save average throughput: 9.7452 Gbps
        np.save (batch version) save average run time: 1.2081 seconds, save average throughput: 19.8683 Gbps
    
    - four hosts:
        np.save (batch version) 
    """
    # Init model and optimizer
    batch_size = 64
    hidden_dim = 8192  # 3072M
    input_dim = output_dim = hidden_dim
    model = MLPModel(hidden_dim=hidden_dim,
                     output_dim=output_dim,
                     manual_pipeline_layer=True)

    # Init batch args
    rngkey = random.PRNGKey(0)
    x = random.normal(rngkey, (batch_size, input_dim), jnp.float32)
    y = jax.random.normal(rngkey, (batch_size, output_dim), jnp.float32)
    batch = {'x': x, 'y': y}

    state = create_train_state(rngkey, model, [x])
    model_size = count_params(state)
    print(f"model size: {model_size * 4 / 1024 / 1024} MB")

    # Compile
    method = PipeshardParallel(num_micro_batches=2)
    parallel_train_step = get_mlp_train_step(method, True, False, False)
    parallel_state = parallel_train_step(state, batch)[0]

    save_tot_duration = 0.0
    save_tot_throughput = 0.0
    outdir = "/home/ubuntu/efs/benchmark_mlp_save"
    cachedir = "/tmp/benchmark_mlp_save"
    for i in range(LOOP_CNT):
        subprocess.run(["rm", "-rf", outdir])
        subprocess.run(["rm", "-rf", cachedir])
        print(f"save to {outdir}")
        # benchmark saving
        start = time.time()
        if i == 0:
            alpa_save_checkpoint("/tmp/warmup", parallel_state, 1)
            jax.block_until_ready(parallel_state)
        else:
            alpa_save_checkpoint(outdir, parallel_state, 1, cachedir)
            #alpa_save_checkpoint("/tmp/warmup", parallel_state, 1)
            jax.block_until_ready(parallel_state)
        duration = time.time() - start
        throughput = model_size * 32 / 1024 / 1024 / 1024 / duration
        if i >= 1:
            save_tot_duration += duration
            save_tot_throughput += throughput
        print(
            f"loop {i} save, time: {duration:.4f} seconds, throughput: {throughput:.4f} Gbps"
        )

    print(
        f"save average run time: {save_tot_duration/(LOOP_CNT - 1):.4f} seconds, save average throughput: {save_tot_throughput/(LOOP_CNT - 1):.4f} Gbps"
    )


def benchmark_mlp_dist_load():
    """
    Benchmark results on local disk:
    - one hosts:
        np.load (batch version) load average run time: 1.6670 seconds, load average throughput: 14.3985 Gbps

    - two hosts:
        TensorStore:            load average run time: 4.4443 seconds, load average throughput: 5.4008 Gbps
        np.load:                load average run time: 3.2214 seconds, load average throughput: 7.4511 Gbps
        np.load (batch version) load average run time: 1.6163 seconds, load average throughput: 14.8510 Gbps
    
    - four hosts:
        np.load (batch version) 
    """
    # Init model and optimizer
    batch_size = 64
    hidden_dim = 8192  # 3072M
    input_dim = output_dim = hidden_dim
    model = MLPModel(hidden_dim=hidden_dim,
                     output_dim=output_dim,
                     manual_pipeline_layer=True)

    # Init batch args
    rngkey = random.PRNGKey(0)
    x = random.normal(rngkey, (batch_size, input_dim), jnp.float32)
    y = jax.random.normal(rngkey, (batch_size, output_dim), jnp.float32)
    batch = {'x': x, 'y': y}

    state = create_train_state(rngkey, model, [x])
    model_size = count_params(state)
    print(f"model size: {model_size * 4 / 1024 / 1024} MB")

    # Compile
    method = PipeshardParallel(num_micro_batches=2)
    parallel_train_step = get_mlp_train_step(method, True, False, False)
    executable = parallel_train_step.get_executable(state, batch)
    state_ss, _ = executable.get_load_info()
    _ = parallel_train_step(state, batch)[0]

    load_tot_duration = 0.0
    load_tot_throughput = 0.0
    outdir = "/tmp/benchmark_mlp_load"
    for i in range(LOOP_CNT):
        print(f"load from {outdir}")
        # benchmark loading
        start = time.time()
        load_state = alpa_restore_checkpoint(outdir, 1, state_ss)
        jax.block_until_ready(load_state)
        duration = time.time() - start
        throughput = model_size * 32 / 1024 / 1024 / 1024 / duration
        if i >= 1:  # first loop for warmup
            load_tot_duration += duration
            load_tot_throughput += throughput
        print(
            f"loop {i} load, time: {duration:.4f} seconds, throughput: {throughput:.4f} Gbps"
        )

    print(
        f"load average run time: {load_tot_duration/(LOOP_CNT - 1):.4f} seconds, load average throughput: {load_tot_throughput/(LOOP_CNT - 1):.4f} Gbps"
    )


if __name__ == "__main__":
    alpa.init(cluster="ray")
    # print("ndarray benchmark on EFS:")
    # print("flax")
    # benchmark_ndarray_save_load(mode="flax")
    # print("\nalpa")
    # benchmark_ndarray_save_load(mode="alpa")
    # print("\nnumpy")
    # benchmark_ndarray_save_load(mode="numpy")

    # print("\n\nndarray benchmark on local disk:")
    # print("flax")
    # benchmark_ndarray_save_load(mode="flax", to_efs=False)
    # print("\nalpa")
    # benchmark_ndarray_save_load(mode="alpa", to_efs=False)
    # print("\nnumpy")
    # benchmark_ndarray_save_load(mode="numpy", to_efs=False)

    # print("mlp benchmark on EFS:")
    # benchmark_mlp_save(mode="flax")
    # benchmark_mlp_save(mode="alpa")
    # benchmark_mlp_save(mode="numpy")

    # print("mlp benchmark on local disk:")
    # benchmark_mlp_save(mode="flax", to_efs=False)
    # benchmark_mlp_save(mode="alpa", to_efs=False)
    # benchmark_mlp_save(mode="numpy", to_efs=False)

    # print("dist array save/load benchmark:")
    # benchmark_dist_arr_save()
    # benchmark_dist_arr_load()

    # print("mlp dist save/load benchmark:")
    # benchmark_mlp_dist_save()
    benchmark_mlp_dist_load()
    alpa.shutdown()

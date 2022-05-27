import os
import subprocess
import time
import timeit
from tempfile import TemporaryFile, TemporaryDirectory

from flax.training.checkpoints import save_checkpoint 
import jax
import jax.numpy as jnp
from jax import random
import numpy as np


import alpa
from alpa import save_checkpoint as alpa_save_checkpoint
from alpa import restore_checkpoint as alpa_restore_checkpoint
from alpa.testing import (MLPModel, BertLayerModel, create_train_state,
                          get_bert_layer_train_step, get_mlp_train_step,
                          assert_allclose)
from alpa.device_mesh import get_global_cluster

def _get_efs_mount_point():
    # Hacky function to get the EFS mount point
    for line in subprocess.check_output("df -h", shell=True).decode().split('\n'):
        cols = line.split(' ')
        if "efs" in cols[0]:
            return cols[-1]+"/"
    return None

def _get_save_prefix():
    device_cluster = get_global_cluster()
    if len(device_cluster.host_info) > 1:
        # Get EFS mount point for the multi-host test
        save_prefix = _get_efs_mount_point()
    else:
        # Use tmp dir for the single-host test
        save_prefix = "/tmp/"
    return save_prefix

LOOP_CNT=5

def benchmark_ndarray_save_load(mode="flax", to_efs=True):
    """
    Save/Load path is set under the EFS filesystem.
    EFS performance: https://docs.aws.amazon.com/efs/latest/ug/performance.html

    if mode == "flax": use flax.training.checkpoints.save_checkpoint/restore_checkpoint
    elif mode == "numpy: use np.save/load

    Benchmark results on EFS: 
    - flax.save_checkpoint: average run time: 1.482362699508667 seconds, average throughput: 0.6749969769443334 Gbps
    - alpa.save_checkpoint: average run time: 1.336690330505371 seconds, average throughput: 0.7484693878347961 Gbps
    - np.save: average run time: 1.2787351608276367 seconds, average throughput: 0.7874443325181828 Gbps

    Benchmark results on local filesystem:
    - flax.save_checkpoint: average run time: 0.6772919178009034 seconds, average throughput: 1.4764804301530936 Gbps
    - alpa.save_checkpoint: average run time: 0.8362024784088135 seconds, average throughput: 1.2280585577116316 Gbps
    - np.save: average run time: 0.09975528717041016 seconds, average throughput: 10.024606837849765 Gbps
    """
    rngkey = random.PRNGKey(0)
    #arr_sizes = [1024*1024, 4*1024*1024, 16*1024*1024, 32*1024*1024] # 4M, 16M, 64M, 128M
    arr_sizes = [32*1024*1024] # 128M
    benchmark_arrs = [random.normal(rngkey, (arr_size,)) for arr_size in arr_sizes]
    for arr in benchmark_arrs:
        tot_duration = 0.0
        tot_throughput = 0.0
        for i in range(LOOP_CNT):
            if to_efs:
                outdir = TemporaryDirectory(prefix=_get_save_prefix()).name
            else:
                outdir = TemporaryDirectory().name
            print (f"save to {outdir}")

            start = time.time()
            if mode == "flax":
                save_checkpoint(outdir, arr, i)
            elif mode == "alpa":
                alpa_save_checkpoint(outdir, arr, i)
            else:
                os.makedirs(outdir, exist_ok=True)
                ckpt_path = os.path.join(outdir, f"checkpoint_1")
                np.save(ckpt_path, arr)
            duration = time.time() - start

            throughput = arr.size * 32 / 1024 / 1024 / 1024 / duration
            tot_duration += duration
            tot_throughput += throughput
            print(f"loop {i}, time: {duration} seconds, throughput: {throughput} Gbps")
        print(f"average run time: {tot_duration/LOOP_CNT} seconds, average throughput: {tot_throughput/LOOP_CNT} Gbps")

def count_params(model):
    return sum(x.size for x in jax.tree_leaves(model))

def benchmark_mlp_save_load(mode="flax", to_efs=True):
    """
    Benchmark results on EFS: 
    - flax.save_checkpoint: average run time: 45.19087886810303 seconds, average throughput: 0.5313484040513637 Gbps
    - alpa.save_checkpoint: average run time: 16.15189399719238, average throughput: 1.4860819837013484 Gbps

    Benchmark results on local disk:
    - flax.save_checkpoint: average run time: 16.1341721534729, average throughput: 1.4877078603042466 Gbps
    - alpa.save_checkpoint: average run time: 10.663438653945922, average throughput: 2.2509621962263244 Gbps
    """
    # Init model and optimizer
    batch_size = 64
    hidden_dim = 8192 # 3072M
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
    for i in range(LOOP_CNT):
        outdir = TemporaryDirectory(prefix=_get_save_prefix()).name
        if to_efs:
            outdir = TemporaryDirectory(prefix=_get_save_prefix()).name
        else:
            outdir = TemporaryDirectory().name

        start = time.time()
        if mode == "flax":
            save_checkpoint(outdir, state, i)
        else:
            alpa_save_checkpoint(outdir, state, i)
        duration = time.time() - start

        throughput = model_size * 32 / 1024 / 1024 / 1024 / duration
        tot_duration += duration
        tot_throughput += throughput
        print(f"loop {i}, time: {duration} seconds, throughput: {throughput} Gbps")
    print(f"average run time: {tot_duration/LOOP_CNT}, average throughput: {tot_throughput/LOOP_CNT} Gbps")



if __name__ == "__main__":
    alpa.init(cluster="ray")
    # print("ndarray benchmark on EFS:")
    # print("flax")
    # benchmark_ndarray_save_load(mode="flax")
    # print("\nalpa")
    # benchmark_ndarray_save_load(mode="alpa")
    # print("\nnumpy")
    # benchmark_ndarray_save_load(mode="numpy")

    print("ndarray benchmark on local disk:") 
    print("flax")
    benchmark_ndarray_save_load(mode="flax", to_efs=False)
    print("\nalpa")
    benchmark_ndarray_save_load(mode="alpa", to_efs=False)
    print("\nnumpy")
    benchmark_ndarray_save_load(mode="numpy", to_efs=False)

    # print("mlp benchmark on EFS:")
    # benchmark_mlp_save_load(mode="flax")
    # benchmark_mlp_save_load(mode="alpa")

    # print("mlp benchmark on local disk:")
    # benchmark_mlp_save_load(mode="flax", to_efs=False)
    # benchmark_mlp_save_load(mode="alpa", to_efs=False)


    alpa.shutdown()
    


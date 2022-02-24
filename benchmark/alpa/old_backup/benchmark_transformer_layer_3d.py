import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

import alpa
from alpa import (parallelize, global_config, set_parallelize_options, DeviceCluster,
                   mark_pipeline, manual_layer_construction)
from alpa.model.bert_model import BertConfig, FlaxBertLayerCollection
from alpa.model.model_util import TrainState
from alpa.util import write_tsv, list_gpu_info, print_used_time, get_ray_namespace_str

MB = 1024 ** 2
GB = 1024 ** 3


def report_pipeline_breakdown(executable, timer_names, niter):
    overall_costs = executable.get_execution_time_costs(warmup=0, timer_name="overall")

    print(">>> overall: {}...".format(overall_costs))
    other_percentage = [100.0] * niter
    other = overall_costs
    for timer_name in timer_names:
        costs = executable.get_execution_time_costs(warmup=0, timer_name=timer_name)
        if len(costs) == 0:
            costs = [0.0] * niter
        percentage = [cost / overall_costs[i] * 100 for i, cost in enumerate(costs)]
        other = [remain - costs[i] for i, remain in enumerate(other)]
        other_percentage = [remain - percentage[i] for i, remain in enumerate(other_percentage)]
        strs = []
        for i, cost in enumerate(costs):
            strs.append(str(cost) + f" ({percentage[i]:.1f}) ")
        print_string = ",".join(strs)
        print(">>> {}: {}".format(timer_name, print_string))

    # print unknown overhead
    strs = []
    for i, remain in enumerate(other):
        strs.append(" " + str(remain) + f" ({other_percentage[i]:.1f})")
    print_string = ",".join(strs)
    print(">>> {}: {}".format("Others: ", print_string))


def create_train_state(rngkey, model, batch):
    params = model.init_dummy(rngkey, batch["hidden_states"], batch["attention_mask"])
    tx = optax.adam(learning_rate=1e-2)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dynamic_scale=None)
    return state


def get_train_step(grad_func, pipeline_mp_size):

    @parallelize
    # @partial(parallelize, donate_argnums=())
    def train_step(state, batch, rng_key):

        def loss_func(params):
            rngs = {"dropout": rng_key}
            if pipeline_mp_size > 1:
                mark_pipeline(name="0", mark_type="start")
            out = state.apply_fn(params,
                                 batch["hidden_states"],
                                 batch["attention_mask"],
                                 deterministic=True,
                                 rngs=rngs)[0]
            loss = jnp.mean((out - batch["label"]) ** 2)
            if pipeline_mp_size > 1:
                mark_pipeline(name=str(pipeline_mp_size - 1), mark_type="end")
            return loss
        if pipeline_mp_size > 1:
            loss_func = manual_layer_construction(loss_func)
        # grad, grad_x = jax.grad(loss_func, argnums=(0, 1))(optimizer.target, batch["hidden_states"])

        params = jax.tree_util.tree_map(lambda x: x, state.params)
        grads = grad_func(loss_func)(params)
        if args.skip_apply_grad:
            return grads
        else:
            new_state = state.apply_gradients(grads=grads)
            return new_state

    return train_step


def benchmark_transformer_one_case(benchmark_case):
    print_used_time(None)

    # Model configs
    batch_size, seq_len, hidden_size, num_layers, num_heads, \
    l_dim0, l_dim1, p_dim0, p_dim1, pipeline_mp_size, num_micro_batches, force_data_parallel, \
    use_remat = benchmark_case


    # do some sanity check
    if l_dim0 > 1 and l_dim1 > 1 and force_data_parallel:
        raise RuntimeError("Force data parallel can only be enabled in 1D logical mesh.")
    if l_dim0 * l_dim1 != p_dim0 * p_dim1:
        raise RuntimeError("logical mesh shape and physical mesh shape are not compatible.")

    global_config.force_data_parallel = force_data_parallel
    global_config.prefer_reduce_scatter = False

    # Control whether we want to do sync more aggressively
    if args.skip_apply_grad and num_micro_batches == 1:
        grad_func = jax.grad
    else:
        grad_func = alpa.grad

    # Mesh configs
    # 3D parallel always run atop a Ray cluster.
    device_cluster = DeviceCluster()
    virtual_mesh = device_cluster.get_virtual_physical_mesh()
    set_parallelize_options(devices=virtual_mesh,
                            strategy="pipeshard_parallel",
                            num_micro_batches=num_micro_batches,
                            sub_physical_mesh_shapes=[(p_dim0, p_dim1)] * pipeline_mp_size,
                            sub_logical_mesh_shapes=[(l_dim0, l_dim1)] * pipeline_mp_size)


    rngkey = jax.random.PRNGKey(0)
    # Prepare input batch
    batch = {
        "hidden_states": jax.random.normal(rngkey, (batch_size, seq_len, hidden_size), dtype=jnp.float32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.float32),
        "label": jax.random.normal(rngkey, (batch_size, seq_len, hidden_size), dtype=jnp.float32)
    }
    print_used_time("Prepare input")

    # Init model and optimizer
    model = FlaxBertLayerCollection(BertConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=num_heads,
        pipeline_mp_size=pipeline_mp_size,
        add_manual_pipeline_markers=True))

    state = create_train_state(rngkey, model, batch)
    print_used_time("Create train state")

    # Compile executable
    train_step = get_train_step(grad_func, pipeline_mp_size)
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    for i in range(args.niter):
        if args.skip_apply_grad:
            train_step(state, batch, rngkey)
        else:
            state = train_step(state, batch, rngkey)

    overall_costs = executable.get_execution_time_costs(warmup=0, timer_name="overall")
    print_used_time("Benchmark")


    report_pipeline_breakdown(executable, ["resharding_send", "resharding_recv", "compute"], args.niter)
    # Log benchmark results
    heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape", "#Microbatch", "Force DP", "Remat", "Mean Time", "Std Time"]
    paralell_config = (benchmark_case[5], benchmark_case[6], benchmark_case[9])
    values = ["transformer-layer", str(benchmark_case[:5]), str(paralell_config),
              str(benchmark_case[7:9]), f"{benchmark_case[10]}", str(benchmark_case[11]), str(benchmark_case[12]),
              f"{np.mean(overall_costs[2:]):.3f}", f"{np.std(overall_costs[2:]):.5f}"]
    write_tsv(heads, values, "result_trans.tsv")

    executable.shutdown()

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers,
# #head = num_heads, LD0 = logical mesh dim 0, LD1 = logical mesh_dimension_1
# PD0 = physical mesh dim 0, PD = physical mesh dim 1
# FD = Force DP, NB = number of microbatches, Remat: rematerialization
benchmark_suite_2_gpu = [
]


benchmark_suite_4_gpu = [
    # B,  S,    H,    L,  #head,     LD0, LD1, PD0, PD1, PP, NB, FD,    Remat,
    (32,  1024, 1536, 2,  1536//96,  1,   2,   1,   2,   2,  1,  False, False),
    (32,  1024, 1536, 2,  1536//96,  1,   2,   1,   2,   2,  2,  False, False),
    (32,  1024, 1536, 2,  1536//96,  1,   2,   1,   2,   2,  4,  False, False),
    (32,  1024, 1536, 2,  1536//96,  1,   2,   1,   2,   2,  8,  False, False),
    (32,  1024, 1536, 2,  1536//96,  1,   2,   1,   2,   2,  16, False, False),

    (32,  1024, 1536, 2,  1536//96,  1,   2,   1,   2,   2,  1,  True, False),
    (32,  1024, 1536, 2,  1536//96,  1,   2,   1,   2,   2,  2,  True, False),
    (32,  1024, 1536, 2,  1536//96,  1,   2,   1,   2,   2,  4,  True, False),
    (32,  1024, 1536, 2,  1536//96,  1,   2,   1,   2,   2,  8,  True, False),
    (32,  1024, 1536, 2,  1536//96,  1,   2,   1,   2,   2,  16, True, False),

    (32,  1024, 1536, 4,  1536//96,  1,   2,   1,   2,   2,  1,  True, False),
    (32,  1024, 1536, 4,  1536//96,  1,   2,   1,   2,   2,  2,  True, False),
    (32,  1024, 1536, 4,  1536//96,  1,   2,   1,   2,   2,  4,  True, False),
    (32,  1024, 1536, 4,  1536//96,  1,   2,   1,   2,   2,  8,  True, False),
    (32,  1024, 1536, 4,  1536//96,  1,   2,   1,   2,   2,  16, True, False), # OOM on Gpipe, but not 1F1B

    (32,  1024, 1536, 4,  1536//96,  1,   2,   1,   2,   2,  1,  False, False), # might OOM
    (32,  1024, 1536, 4,  1536//96,  1,   2,   1,   2,   2,  2,  False, False),
    (32,  1024, 1536, 4,  1536//96,  1,   2,   1,   2,   2,  4,  False, False),
    (32,  1024, 1536, 4,  1536//96,  1,   2,   1,   2,   2,  8,  False, False),
    (32,  1024, 1536, 4,  1536//96,  1,   2,   1,   2,   2,  16, False, False), # Gpipe OOM
    (32,  1024, 1536, 4,  1536//96,  1,   2,   1,   2,   2,  32, False, False), # might OOM

    (32,  1024, 1536, 4,  1536//96,  1,   1,   1,   1,   4,  1,  False, False), # might OOM
    (32,  1024, 1536, 4,  1536//96,  1,   1,   1,   1,   4,  2,  False, False),
    (32,  1024, 1536, 4,  1536//96,  1,   1,   1,   1,   4,  4,  False, False),
    (32,  1024, 1536, 4,  1536//96,  1,   1,   1,   1,   4,  8,  False, False),
    (32,  1024, 1536, 4,  1536//96,  1,   1,   1,   1,   4,  16, False, False),
    (32,  1024, 1536, 4,  1536//96,  1,   1,   1,   1,   4,  32, False, False),

    # memory stress tests
    (512,  1024, 1536, 4,  1536//96,  1,   1,   1,   1,  4,  128, False, False),
]

benchmark_suite_8_gpu = [

]


def benchmark_all(use_profiling):
    if args.local:
        num_gpus = list_gpu_info().count("UUID")
    else:
        num_gpus = int(ray.cluster_resources()["GPU"])

    benchmark_suites = {
        2: benchmark_suite_2_gpu,
        4: benchmark_suite_4_gpu,
        8: benchmark_suite_8_gpu,
    }
    print(">>> num_gpus: ", num_gpus)
    for case in benchmark_suites[num_gpus]:
        # Backup global config
        backup = global_config.backup()
        benchmark_transformer_one_case(case)
        # Restore global config
        global_config.restore(backup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--local", action="store_true",
                        help="Run on local GPUs. Do not use ray actors.")
    parser.add_argument("--niter", type=int, default=10,
                        help="Number of benchmark iteration")
    parser.add_argument("--skip-apply-grad", type=bool, default=False,
                        help="Whether we want to skip applying the gradients")
    args = parser.parse_args()

    if not args.local:
        ray.init(address="auto", namespace=get_ray_namespace_str())
        jax.config.update('jax_platform_name', 'cpu')

    global_config.use_dummy_value_for_benchmarking = True

    benchmark_all(args.use_profiling)

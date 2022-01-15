"""Project tflops for exp_intra_only.py"""

def project_tflops(micro_batch_size, batch_sizes, latencies, tflops, queries):
    per_nb_latency = (latencies[1] - latencies[0]) / ((batch_sizes[1] - batch_sizes[0]) / micro_batch_size)
    base_latency = latencies[0] - (batch_sizes[0] / micro_batch_size) * per_nb_latency

    ret_latencies = []
    ret_tflops = []
    for query_bs in queries:
        nb = query_bs / micro_batch_size
        cur_latency = nb * per_nb_latency + base_latency
        cur_tflops = (latencies[0] * tflops[0] * query_bs / batch_sizes[0]) / cur_latency

        ret_latencies.append(round(cur_latency, 4))
        ret_tflops.append(round(cur_tflops, 4))

    return ret_latencies, ret_tflops

# gpt - 4 nodes
ret_latencies, ret_tflops = project_tflops(
    32, [64, 128], [39.09, 42.525], [6.6104, None], queries=[64, 128, 256, 1024])
print("gpt", ret_latencies, ret_tflops)

# moe - 4 nodes
ret_latencies, ret_tflops = project_tflops(
    32, [64, 128], [31.5075, 59.4105], [1.0895, None], queries=[64, 128, 256, 1024])
print("moe", ret_latencies, ret_tflops)

# wresnet - 4 nodes
ret_latencies, ret_tflops = project_tflops(
    32, [64, 128], [32.6111, 54.1155], [0.3807, None], queries=[64, 128, 256, 1536])
print("wresnet", ret_latencies, ret_tflops)



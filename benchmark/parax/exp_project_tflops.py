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
latencies, tflops = project_tflops(
    32, [128, 256], [42.525, 53.2671], [12.1367, None], queries=[64, 128, 256, 1024])
print(f"gpt 4 node, latency: {latencies}, tflops: {tflops}")

# moe - 4 nodes
latencies, tflops = project_tflops(
    32, [128, 256], [50.6981, 95.2731], [1.3457, None], queries=[64, 128, 256, 1024])
print(f"moe 4 node, latency: {latencies}, tflops: {tflops}")

# moe - 8 nodes
latencies, tflops = project_tflops(
    32, [128, 256], [157.1711, 319.2686], [0.221, None], queries=[64, 128, 256, 1024])
print(f"moe 8 node, latency: {latencies}, tflops: {tflops}")

# wresnet - 4 nodes
latencies, tflops = project_tflops(
    32, [128, 256], [43.3694, 53.4454], [0.582, None], queries=[64, 128, 256, 1536])
print(f"wresnet 4 node, latency: {latencies}, tflops: {tflops}")

# wresnet - 8 nodes
latencies, tflops = project_tflops(
    8, [32, 64], [41.1547, 49.596], [0.1496, None], queries=[16, 32, 64, 1536])
print(f"wresnet 8 node, latency: {latencies}, tflops: {tflops}")

"""Options of a benchmark case."""
from collections import namedtuple

BenchmarkCase = namedtuple("BenchmarkCase", [
    "batch_size", "model_config", "num_micro_batches", "parallel_mode",
    "parallel_args"
])

ShardParallelArgs = namedtuple(
    "ShardParallelArgs", [("prefer_reduce_scatter", "use_remat",
                           "logical_mesh_shape", "force_batch_dim_mapping")])

UniformParallelArgs = namedtuple("UniformParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "dp", "op", "pp",
    "force_batch_dim_mapping"
])

SearchParallelArgs = namedtuple("SearchParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "num_auto_layers", "auto_stage_option"
])

LoadSolutionParallelArgs = namedtuple("SearchParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "num_auto_layers",
    "forward_stage_layer_ids", "submesh_physical_shapes",
    "submesh_logical_shapes", "submesh_autosharding_option_dicts"
])

"""All global configurations for this project."""


class GlobalConfig:
    """Global configuration of parax."""

    def __init__(self):
        # choices: {'data_parallel', 'auto_sharding'}
        self.shard_parallel_strategy = 'auto_sharding'


global_config = GlobalConfig()

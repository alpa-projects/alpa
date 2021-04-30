"""All global configurations for this project"""

class GlobalConfig:
    def __init__(self):
        # choices: {'data_parallel', 'auto_sharding'}
        self.shard_parallel_strategy = 'data_parallel'

        # choices: {'normal', 'force_data_parallel'}
        self.auto_sharding_solver_strategy = 'normal'

global_config = GlobalConfig()


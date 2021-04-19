"""Global configuration"""


class GlobalConfig:
    def __init__(self):
        self.shard_parallel_strategy = 'data_parallel'

config = GlobalConfig()

def shard_parallel_strategy():
    return config.shard_parallel_strategy

def set_shard_parallel_strategy(strategy):
    global config
    config.shard_parallel_strategy = strategy


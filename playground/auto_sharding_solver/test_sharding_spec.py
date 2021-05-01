from hlo import ShardingSpec
from cluster_env import ClusterEnvironment


def test_tile():
    cluster_env = ClusterEnvironment([[0, 1], [2,3]], None)

    sharding = ShardingSpec.tile((8, 8), [0, 1], [0, 1], cluster_env)
    assert sharding.tile_assignment_dimensions == (2, 2)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3)
    assert sharding.replicate_on_last_tile_dim == False

    sharding = ShardingSpec.tile((8, 8), [1, 0], [1, 0], cluster_env)
    assert sharding.tile_assignment_dimensions == (2, 2)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3)
    assert sharding.replicate_on_last_tile_dim == False

    sharding = ShardingSpec.tile((8, 8), [0, 1], [1, 0], cluster_env)
    assert sharding.tile_assignment_dimensions == (2, 2)
    assert sharding.tile_assignment_devices == (0, 2, 1, 3)
    assert sharding.replicate_on_last_tile_dim == False

    sharding = ShardingSpec.tile((8, 8), [0], [0], cluster_env)
    assert sharding.tile_assignment_dimensions == (2, 1, 2)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3)
    assert sharding.replicate_on_last_tile_dim == True

    sharding = ShardingSpec.tile((8, 8), [0], [1], cluster_env)
    assert sharding.tile_assignment_dimensions == (2, 1, 2)
    assert sharding.tile_assignment_devices == (0, 2, 1, 3)
    assert sharding.replicate_on_last_tile_dim == True

    sharding = ShardingSpec.tile((8, 8), [1], [1], cluster_env)
    assert sharding.tile_assignment_dimensions == (1, 2, 2)
    assert sharding.tile_assignment_devices == (0, 2, 1, 3)
    assert sharding.replicate_on_last_tile_dim == True

    sharding = ShardingSpec.tile((8, 8), [1], [0], cluster_env)
    assert sharding.tile_assignment_dimensions == (1, 2, 2)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3)
    assert sharding.replicate_on_last_tile_dim == True

    sharding = ShardingSpec.tile((8, 8, 8), [0, 1], [0, 1], cluster_env)
    assert sharding.tile_assignment_dimensions == (2, 2, 1)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3)
    assert sharding.replicate_on_last_tile_dim == False

    sharding = ShardingSpec.tile((8, 8, 8), [1], [0], cluster_env)
    assert sharding.tile_assignment_dimensions == (1, 2, 1, 2)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3)
    assert sharding.replicate_on_last_tile_dim == True

if __name__ == "__main__":
    test_tile()


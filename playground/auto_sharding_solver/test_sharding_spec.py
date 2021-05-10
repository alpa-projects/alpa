from hlo import ShardingSpec, ShardingSpecType
from cluster_env import ClusterEnvironment
from common import compute_bytes


def test_tile():
    cluster_env = ClusterEnvironment([[0, 1, 2], [3, 4, 5]], [1,1], [1,1], None)

    sharding = ShardingSpec.tile((12, 12), [0, 1], [0, 1], cluster_env)
    assert sharding.tile_assignment_dimensions == (2, 3)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3, 4, 5)
    assert sharding.replicate_on_last_tile_dim == False

    sharding = ShardingSpec.tile((12, 12), [1, 0], [1, 0], cluster_env)
    assert sharding.tile_assignment_dimensions == (2, 3)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3, 4, 5)
    assert sharding.replicate_on_last_tile_dim == False

    sharding = ShardingSpec.tile((12, 12), [0, 1], [1, 0], cluster_env)
    assert sharding.tile_assignment_dimensions == (3, 2)
    assert sharding.tile_assignment_devices == (0, 3, 1, 4, 2, 5)
    assert sharding.replicate_on_last_tile_dim == False

    sharding = ShardingSpec.tile((12, 12), [0], [0], cluster_env)
    assert sharding.tile_assignment_dimensions == (2, 1, 3)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3, 4, 5)
    assert sharding.replicate_on_last_tile_dim == True

    sharding = ShardingSpec.tile((12, 12), [0], [1], cluster_env)
    assert sharding.tile_assignment_dimensions == (3, 1, 2)
    assert sharding.tile_assignment_devices == (0, 3, 1, 4, 2, 5)
    assert sharding.replicate_on_last_tile_dim == True

    sharding = ShardingSpec.tile((12, 12), [1], [1], cluster_env)
    assert sharding.tile_assignment_dimensions == (1, 3, 2)
    assert sharding.tile_assignment_devices == (0, 3, 1, 4, 2, 5)
    assert sharding.replicate_on_last_tile_dim == True

    sharding = ShardingSpec.tile((12, 12), [1], [0], cluster_env)
    assert sharding.tile_assignment_dimensions == (1, 2, 3)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3, 4, 5)
    assert sharding.replicate_on_last_tile_dim == True

    sharding = ShardingSpec.tile((12, 12, 12), [0, 1], [0, 1], cluster_env)
    assert sharding.tile_assignment_dimensions == (2, 3, 1)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3, 4, 5)
    assert sharding.replicate_on_last_tile_dim == False

    sharding = ShardingSpec.tile((12, 12, 12), [0, 1], [1, 0], cluster_env)
    assert sharding.tile_assignment_dimensions == (3, 2, 1)
    assert sharding.tile_assignment_devices == (0, 3, 1, 4, 2, 5)
    assert sharding.replicate_on_last_tile_dim == False

    sharding = ShardingSpec.tile((12, 12, 12), [1], [0], cluster_env)
    assert sharding.tile_assignment_dimensions == (1, 2, 1, 3)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3, 4, 5)
    assert sharding.replicate_on_last_tile_dim == True


def test_tile2():
    cluster_env = ClusterEnvironment([[0, 1, 2, 3]], [1,1], [1,1], None)
    sharding = ShardingSpec.tile((12, 12), [1], [1], cluster_env)
    assert sharding.tile_assignment_dimensions == (1, 4)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3)
    assert sharding.replicate_on_last_tile_dim == False

    sharding = ShardingSpec.tile((12, 12), [1], [0], cluster_env)
    assert sharding.type == ShardingSpecType.REPLICATED

    cluster_env = ClusterEnvironment([[0], [1], [2], [3]], [1,1], [1,1], None)
    sharding = ShardingSpec.tile((12, 12), [1], [0], cluster_env)
    assert sharding.tile_assignment_dimensions == (1, 4)
    assert sharding.tile_assignment_devices == (0, 1, 2, 3)
    assert sharding.replicate_on_last_tile_dim == False

    sharding = ShardingSpec.tile((12, 12), [1], [1], cluster_env)
    assert sharding.type == ShardingSpecType.REPLICATED


def test_tile3():
    cluster_env = ClusterEnvironment([[0, 1], [2, 3]], [1,1], [1,1], None)
    shape = (12, 12)
    src = ShardingSpec.split(shape, 1, cluster_env)
    dst = ShardingSpec.tile(shape, [0], [0], cluster_env)

    print(src)
    print(dst)
    cost = cluster_env.resharding_cost(shape, src, dst)

    print(cost)


def assert_allclose(x, y):
    assert abs((x - y) / (y + 1e-8))  < 0.01


def test_resharding_cost():
    cluster_env = ClusterEnvironment([[0, 1, 2], [3, 4, 5]], [1, 1], [1, 1], None)
    shape = (128, 128)

    src = ShardingSpec.tile(shape, [0], [0], cluster_env)
    dst = ShardingSpec.tile(shape, [0], [0], cluster_env)
    cost = cluster_env.resharding_cost(shape, src, dst)
    assert_allclose(cost, 0)

    src = ShardingSpec.tile(shape, [0, 1], [0, 1], cluster_env)
    dst = ShardingSpec.tile(shape, [1, 0], [1, 0], cluster_env)
    cost = cluster_env.resharding_cost(shape, src, dst)
    assert_allclose(cost, 0)

    src = ShardingSpec.tile(shape, [0], [0], cluster_env)
    dst = ShardingSpec.tile(shape, [0, 1], [0, 1], cluster_env)
    cost = cluster_env.resharding_cost(shape, src, dst)
    assert_allclose(cost, 0)

    src = ShardingSpec.tile(shape, [0], [0], cluster_env)
    dst = ShardingSpec.tile(shape, [0, 1], [0, 1], cluster_env)
    cost = cluster_env.resharding_cost(shape, src, dst)
    assert_allclose(cost, 0)

    src = ShardingSpec.tile(shape, [0, 1], [0, 1], cluster_env)
    dst = ShardingSpec.tile(shape, [0], [0], cluster_env)
    cost = cluster_env.resharding_cost(shape, src, dst)
    assert_allclose(cost, cluster_env.all_gather_cost(compute_bytes(shape), 1))

    src = ShardingSpec.tile(shape, [0, 1], [0, 1], cluster_env)
    dst = ShardingSpec.replicated(cluster_env)
    cost = cluster_env.resharding_cost(shape, src, dst)
    assert_allclose(cost, cluster_env.all_gather_cost(compute_bytes(shape), 0)
                        + cluster_env.all_gather_cost(compute_bytes(shape), 1))


def test_resharding_cost2():
    cluster_env = ClusterEnvironment([[0], [1], [2], [3]], [1,1], [1,1], None)
    shape = (128, 128)

    src = ShardingSpec.tile(shape, [0, 1], [0, 1], cluster_env)
    dst = ShardingSpec.tile(shape, [0], [0], cluster_env)
    cost = cluster_env.resharding_cost(shape, src, dst)
    assert_allclose(cost, 0)


if __name__ == "__main__":
    test_tile()
    test_tile2()
    #test_tile3()
    test_resharding_cost()
    test_resharding_cost2()


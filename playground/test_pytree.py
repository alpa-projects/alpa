import jax


a = (0, 1, (1, 0))
args, in_tree = jax.tree_util.tree_flatten(a)

print(jax.api_util.flatten_axes("test_pytree", in_tree, 0))

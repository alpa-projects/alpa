import jax
from jax import random, numpy as jnp


class DataLoader:
    """A synthetic data loader"""

    def __init__(self, batch_size, hidden_size, n_batch):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_batch = n_batch
        self.key = random.PRNGKey(0)

    def __iter__(self):
        self.ct = 0
        return self

    def __next__(self):
        if self.ct == self.n_batch:
            raise StopIteration

        self.ct += 1

        return (random.uniform(self.key, shape=(self.batch_size, self.hidden_size)),
                random.uniform(self.key, shape=(self.batch_size, self.hidden_size)))


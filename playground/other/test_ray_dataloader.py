import ray
import jax

import input_pipeline

@ray.remote
class Worker:
    def __init__(self):
        self.generator = None

    def register_generator(self, func):
        self.generator = iter(func())

    def get_next(self):
        return next(self.generator)


def make_generator():
    import tensorflow as tf
    import tensorflow_datasets as tfds

    dataset_builder = tfds.builder('imagenet2012:5.*.*')
    batch_size = 64
    image_size = 224
    dtype = tf.float32
    train = True
    cache = True

    ds = input_pipeline.create_split(
        dataset_builder, batch_size, image_size=image_size, dtype=dtype,
        train=train, cache=cache)
    it = map(lambda xs: jax.tree_map(lambda x: x._numpy(), xs), ds)
    return it


if __name__ == "__main__":
    ray.init(address="auto")

    worker = Worker.remote()

    worker.register_generator.remote(make_generator)

    x = ray.get(worker.get_next.remote())
    print(x.keys())
    print(x['image'].shape)

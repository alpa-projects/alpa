import ray
import jax

@ray.remote
class Actor:
    def __init__(self, name):
        self.name = name
        print(self.name, jax.devices())

ray.init("auto")

actor_1 = Actor.remote("actor_1")
actor_2 = Actor.remote("actor_2")

print(ray.get([Actor.remote("A"), Actor.remote("B")]))
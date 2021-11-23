import ray
import os
import jax


@ray.remote(runtime_env={
    "env_vars": {
        "CUDA_VISIBLE_DEVICS": "0,1,2,3,4,5,6,7",
        "ANOTHER_ENVVAR": "abcd"
    }
})
class Actor:
    def __init__(self, name):
        self.name = name
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        print(self.name, jax.devices(), os.getenv("CUDA_VISIBLE_DEVICES"), os.getenv("ANOTHER_ENVVAR"))

    def get_name(self):
        return self.name

ray.init("auto")

actor_1 = Actor.remote("actor_1")
actor_2 = Actor.remote("actor_2")

ray.get([actor_1.get_name.remote(), actor_2.get_name.remote()])


class LocalActor:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


@ray.remote
class WrappedActor:
    def __init__(self, actor_name):
        self.actor = LocalActor(actor_name)
        for attr in dir(self.actor):
            if not attr.startswith("__"):
                setattr(self, attr, getattr(self.actor, attr))

w = WrappedActor.remote("wrapped_actor")
print(ray.get(w.get_name.remote()))

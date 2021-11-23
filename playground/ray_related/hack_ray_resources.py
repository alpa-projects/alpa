import ray
import os
import jax

@ray.remote(runtime_env={
    "env_vars": {
        "CUDA_VISIBLE_DEVICS": "0,1,2,3,4,5,6,7",
        "CUDA_VISIBLE_DEVICESS": "abcd"
    }
})
class Actor:
    def __init__(self, name):
        self.name = name
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        print(self.name, jax.devices(), os.getenv("CUDA_VISIBLE_DEVICES"), os.getenv("CUDA_VISIBLE_DEVICESS"))

    def get_name(self):
        return self.name

ray.init("auto")

actor_1 = Actor.remote("actor_1")
actor_2 = Actor.remote("actor_2")

ray.get([actor_1.get_name.remote(), actor_2.get_name.remote()])

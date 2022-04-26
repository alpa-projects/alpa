import ray
import time

ray.init(address='auto')

num_gpus = ray.cluster_resources()['GPU']
print('#GPU:', num_gpus)

while num_gpus != 32:
    print("Wait...")
    time.sleep(10)
    num_gpus = ray.cluster_resources()['GPU']
    print('#GPU:', num_gpus)

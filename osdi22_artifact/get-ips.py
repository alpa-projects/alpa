import ray
import socket
ray.init(address="auto")
d = ray.cluster_resources()
my_addr = socket.gethostbyname(socket.gethostname())
for k in d:
    if k.startswith('node'):
        ip = k.split(':')[1]
        print(ip)

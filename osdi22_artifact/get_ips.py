import ray
import socket
ray.init(address="auto")
d = ray.cluster_resources()

# Make sure the head node comes at first
my_addr = socket.gethostbyname(socket.gethostname())
print("My IP address: {}".format(my_addr))
with open("ips", "w") as ips:
    ips.write(my_addr + "\n")
    for k in d:
        if k.startswith('node'):
            ip = k.split(':')[1]
            if ip == my_addr:
                continue
            ips.write(ip + "\n")
            print(ip)

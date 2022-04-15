import ray
import socket
ray.init(address="auto")
d = ray.cluster_resources()

# Make sure the head node comes at first
my_addr = socket.gethostbyname(socket.gethostname())
print("My IP address: {}".format(my_addr))

ips = []
for k in d:
    if k.startswith('node'):
        ip = k.split(':')[1]
        if ip == my_addr:
            continue
        ips.append(ip)
        print(ip)

with open("deepspeed/hostfile", "w") as hostfile:
    hostfile.write(my_addr + " slots=8")

with open("deepspeed/hostfile_2node", "w") as hostfile:
    hostfile.write(my_addr + " slots=8\n")
    hostfile.write(ips[0] + " slots=8")

with open("deepspeed/hostfile_4node", "w") as hostfile:
    hostfile.write(my_addr + " slots=8\n")
    hostfile.write(ips[0] + " slots=8\n")
    hostfile.write(ips[1] + " slots=8\n")
    hostfile.write(ips[2] + " slots=8")

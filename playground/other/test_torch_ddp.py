"""
Usage:
python3 -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 11000 test_torch_ddp.py
"""
import torch
import torch.optim as optim
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
#from torch.nn.parallel import DataParallel as torchDDP

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.net1 = nn.Linear(1 << 10, 1 << 19)
        self.net2 = nn.Linear(1 << 19, 1)

    def forward(self, x):
        return self.net2(self.net1(x))


GB = 1024 ** 3

def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    rank = torch.distributed.get_rank()
    device = rank % torch.cuda.device_count()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    if print_info:
        print("allocated: %.2f GB" % (allocated / GB), flush=True)
        print("reserved:  %.2f GB" % (reserved / GB), flush=True)
    return allocated

torch.distributed.init_process_group(backend="nccl", world_size=1)

raw_model = Net().cuda()

print("After init model", get_memory_usage() / GB)
model = torchDDP(raw_model, device_ids=[0], output_device=0, gradient_as_bucket_view=True)
optimizer = optim.SGD(model.parameters(), lr=0.001)

print("After torchDDP", get_memory_usage() / GB)

data = torch.ones((1, 1<<10)).cuda()
label = torch.ones((1,)).cuda()

optimizer.zero_grad()
loss = torch.square(model(data) - label).sum()
loss.backward()
optimizer.step()

print("After first backward", get_memory_usage() / GB)

optimizer.zero_grad()
loss = torch.square(model(data) - label).sum()
loss.backward()
optimizer.step()
print("After second backward", get_memory_usage() / GB)


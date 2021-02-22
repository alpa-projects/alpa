import torch
from torch import nn


def auto_parallel(func, args):
    graph = torch.jit._get_trace_graph(func, args=args)

    print(graph)


class Model(nn.Module):
    def __init__(self, H, V):
        super().__init__()
        self.l1 = nn.Linear(H, H)
        self.l2 = nn.Linear(H, V)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = nn.functional.log_softmax(x, dim=-1)

        return x


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size, "%d vs %d" % (x.size(1), self.size)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        true_dist.requires_grad_(False)
        return self.criterion(x, true_dist)


N = 16
S = 512
H = 1024
V = 4096

class LossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model(H, V)
        self.criterion = LabelSmoothing(V, 0, 0.1)

    def forward(self, data, target):
        output = self.model(data)
        #loss = self.criterion(output.view(-1, V), target.view(-1))
        return output

data = torch.randn(N, S, H)
target = torch.randint(0, V, (N, S))

#train_func(data, target)

#graph = torch.jit._get_trace_graph(train_func, (data, target))

graph = torch.jit.trace(LossWrapper(), (data, target))
#LossWrapper()(data, target)


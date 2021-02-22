import torch

N = 2
H = 4

loss_func = torch.nn.MSELoss()
model = torch.nn.Linear(H, H)

def func(data, target, *params):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    y = model(data)
    loss = loss_func(y, target)

    print(y)

    loss.backward()
    return loss

data = torch.ones((N, H))
target = torch.ones((N, H))

model_params = tuple(model.parameters())
func(*((data, target,) + model_params))
model_grads = tuple(x.grad for x in model_params)

graph, output = torch.jit._get_trace_graph(func, (data, target) + model_params + model_grads)

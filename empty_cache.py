import torch
from torch._C import device


x = torch.tensor([[1., -1.], [1., 1.]]).cuda().requires_grad_()
out = x.pow(2).sum()
out.backward()
print(x.grad)
torch.cuda.empty_cache()
x = torch.tensor([[2., -1.], [1., 1.]]).cuda().requires_grad_()
print(x)
print(x.grad)
input(0)


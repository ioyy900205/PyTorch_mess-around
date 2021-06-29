import torch
from torch import tensor
mid_1 = torch.FloatTensor(1)
# print(mid_1)
_p = mid_1.fill_(19 * 1.0 / 20)
# print(_p)
# _p = torch.tensor([0.0500])
output_1 = torch.log(_p)
# print(output_1)

c = torch.range(2,5)
# print(c)

c_2 = c*output_1
print(c_2)

c_3 = torch.exp(c_2)
# print(c_3)

c_3 /= c_3.sum()
# print(c_3,c_3.sum())

mid_1 = torch.randn(5,5)
# print(mid_1)
max_preds, argmax_preds = mid_1.max(dim=0, keepdim=True)
# print(max_preds)
# print(argmax_preds)
print(c_2.sort(descending=True))
filtered = torch.zeros(9)
cp_1 = torch.arange(1,10)
c = cp_1.ge(3)
print(cp_1,c)
print(filtered)
filtered.add_(cp_1.ge(3).type_as(filtered))
print(filtered)
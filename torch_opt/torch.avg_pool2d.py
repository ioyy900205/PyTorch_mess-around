import torch
from torch.nn import functional as F

# 1.初始化
input = torch.tensor(
    [
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,0,0,1,1],
        [1,1,1,1,1]
        
    ]).unsqueeze(0).float()
# print(input.size())#torch.Size([1, 4, 5])
# print(input)

#2. avg_pool1d
# m1 = F.avg_pool1d(input,kernel_size=2)
# print(m1)#tensor([[[1.0000, 1.0000],
#          # [1.0000, 1.0000],
#          # [0.0000, 0.5000],
#          # [1.0000, 1.0000]]])
 
#3. avg_pool2d
# m = F.avg_pool2d(input,kernel_size=2)#这里是在原矩阵中找出2*2的区域求平均，不够的舍弃，stride=(2,2)
# print(m)#tensor([[[1.0000, 1.0000],
#          # [0.5000, 0.7500]]])


# m3= F.avg_pool2d(input,kernel_size=2,stride=1)#这里是在原矩阵中找出2*2的区域求平均，不够的舍弃，stride=(1,1),[1,1,4,5]-->[1,1,3,4]
# print(m3)#tensor([[[1.0000, 1.0000, 1.0000, 1.0000],
#          # [0.5000, 0.5000, 0.7500, 1.0000],
#          # [0.5000, 0.5000, 0.7500, 1.0000]]])

c = torch.randn(256,384,4,4)
print(c.size())
out = F.avg_pool2d(c, 4)
print(out.size())
print("out.size(0)",out.size(0))
out = out.view(out.size(1),-1)
out.size()
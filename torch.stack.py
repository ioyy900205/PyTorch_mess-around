'''
Date: 2021-05-10 11:49:32
LastEditors: Liuliang
LastEditTime: 2021-05-11 14:15:07
Description: 
'''
import torch
import numpy as np

# 创建3*3的矩阵，a、b
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
b=np.array([[10,20,30],[40,50,60],[70,80,90]])
# 将矩阵转化为Tensor
a = torch.from_numpy(a)
b = torch.from_numpy(b)
# 打印a、b、c
print(a,a.size())
print(b,b.size())


d = torch.stack((a, b), dim=0)
print(d)
print(d.size())


d = torch.stack((a, b), dim=1)
print(d)
print(d.size())

d = torch.stack((a, b), dim=2)
print(d)
print(d.size())



'''
Date: 2021-05-21 15:43:43
LastEditors: Liuliang
LastEditTime: 2021-05-21 15:50:36
Description: view
'''
import torch
a = torch.arange(0,12,1)

print(a)

b = a.view(2,-1)

print(b)

c = b.view(6,2)

print(c)

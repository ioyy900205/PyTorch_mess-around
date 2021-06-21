'''
Date: 2021-06-17 11:57:01
LastEditors: Liuliang
LastEditTime: 2021-06-17 12:17:23
Description: 
'''
import torch
import torch.tensor as tensor

loss = (tensor(4.4589, device='cuda:1'), tensor(3.9323, device='cuda:1'))

loss_aux = (tensor(4.4589, device='cuda:1'), tensor(3.9323, device='cuda:1'))



for l1,l2 in loss,loss_aux:
    c = l1+l2

print(c)
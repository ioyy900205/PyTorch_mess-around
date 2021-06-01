'''
Date: 2021-05-31 19:55:54
LastEditors: Liuliang
LastEditTime: 2021-05-31 20:02:32
Description: flatten
'''
import torch
input = torch.randn(32,3,9,9)
print(input.size())
input_2 = input.flatten(start_dim = 2) 
print(input_2.size())
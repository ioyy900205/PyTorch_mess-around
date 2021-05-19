'''
Date: 2021-05-19 09:50:09
LastEditors: Liuliang
LastEditTime: 2021-05-19 09:51:25
Description: 
'''
#叶子节点可以理解成不依赖其他tensor的tensor
import torch 
a=torch.tensor([1.0])
print(a.is_leaf)

b=a+1
print(b.is_leaf)
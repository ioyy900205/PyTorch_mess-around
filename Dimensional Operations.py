import torch
A = torch.ones(2,3) #2x3的张量（矩阵） 
print("A:",A)
B=2*torch.ones(4,3) #4x3的张量（矩阵） 
print("B:",B)
C=torch.cat((A,B),0)#按维数0（行）拼接
print("C:",C)


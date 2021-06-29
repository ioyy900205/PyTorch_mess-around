import torch

# 二维数组
A = torch.ones(2,3) #2x3的张量（矩阵） 
print("A:",A)
B=2*torch.ones(4,3) #4x3的张量（矩阵） 
print("B:",B)
C=torch.cat((A,B),0)#按维数0（行）拼接
print("C:",C)
print(C.size())

#二维数组dim=1
A = torch.ones(3,2) #2x3的张量（矩阵） 
print("A:",A)
B=2*torch.ones(3,4) #4x3的张量（矩阵） 
print("B:",B)
C=torch.cat((A,B),1)#按维数1（行）拼接
print("C:",C)
print(C.size())

# 三维数组 dim=0
A = torch.ones(2,2,3) #2x2x3的张量（矩阵） 
print("A:",A)
B=2*torch.ones(4,2,3) #4x3的张量（矩阵） 
print("B:",B)
C=torch.cat((A,B),0)#按维数0（行）拼接
print("C:",C)
print(C.size())

# 三维数组 dim=1
A = torch.ones(2,2,3) #2x2x3的张量（矩阵） 
print("A:",A)
B=2*torch.ones(2,4,3) #4x3的张量（矩阵） 
print("B:",B)
C=torch.cat((A,B),1)#按维数1（行）拼接
print("C:",C)
print(C.size())

# 三维数组 dim=2
A = torch.ones(2,3,2) #2x2x3的张量（矩阵） 
print("A:",A)
B=2*torch.ones(2,3,4) #4x3的张量（矩阵） 
print("B:",B)
C=torch.cat([A,B],2)#按维数 2 拼接
print("C:",C)
print(C.size())

torch.manual_seed(0)

x_2_input = torch.randn(8,3,24,24)
c = x_2_input[7]
print(c)
print(c.size())


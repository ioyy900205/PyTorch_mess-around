import torch


device = torch.device('cuda:0')
# # 定义两个tensor
dummy_tensor_4 = torch.randn(100, 3, 512, 512).float().to(device)  #885  300M+ 585M
dummy_tensor_5 = torch.randn(100, 3, 512, 512).float().to(device)  #1185 300M
dummy_tensor_6 = torch.tensor(dummy_tensor_5,dtype = torch.int64).to(device)  #1485 300M

# dummy_tensor_7 = torch.randn(100, 3, 512, 512).int().to(device) #885
# # dummy_tensor_8 = torch.randn(100, 3, 512, 512).int().to(device) #1185 300M


# dummy_tensor_9 =torch.tensor(dummy_tensor_7, dtype = torch.int8).to(device) #961  76M
# dummy_tensor_10 =torch.tensor(dummy_tensor_7, dtype = torch.int16).to(device)# 1111 150M
# dummy_tensor_11 =torch.tensor(dummy_tensor_7, dtype = torch.int32).to(device)# 1411 300M
# dummy_tensor_12 =torch.tensor(dummy_tensor_7, dtype = torch.int64).to(device)# 2011 600M


import numpy as np
import torch
import sys

# 32位整型
ai32 = torch.tensor([], dtype=torch.int32)
bi32 = torch.tensor(1, dtype=torch.int32)
ci32 = torch.tensor(5, dtype=torch.int32)

# 64位整型
ai64 = torch.tensor([], dtype=torch.int64)
bi64 = torch.tensor(1, dtype=torch.int64)
ci64 = torch.tensor(5, dtype=torch.int64)

# 32位浮点数
af32 = torch.tensor([], dtype=torch.float32)
bf32 = torch.tensor(1, dtype=torch.float32)
cf32 = torch.tensor(5, dtype=torch.float32)

# 64位浮点数
af64 = torch.tensor([], dtype=torch.float64)
bf64 = torch.tensor(1, dtype=torch.float64)
cf64 = torch.tensor(5, dtype=torch.float64)

print("size of 0 int32 number: %f" % sys.getsizeof(ai32))
print("size of 1 int32 number: %f" % sys.getsizeof(bi32))
print("size of 5 int32 numbers: %f" % sys.getsizeof(ci32), end='\n\n')

print("size of 0 int64 number: %f" % sys.getsizeof(ai64))
print("size of 1 int64 number: %f" % sys.getsizeof(bi64))
print("size of 5 int64 numbers: %f" % sys.getsizeof(ci64), end='\n\n')

print("size of 0 float32 number: %f" % sys.getsizeof(af32))
print("size of 1 float32 number: %f" % sys.getsizeof(bf32))
print("size of 5 float32 numbers: %f" % sys.getsizeof(cf32), end='\n\n')

print("size of 0 float64 number: %f" % sys.getsizeof(af64))
print("size of 1 float64 number: %f" % sys.getsizeof(bf64))
print("size of 5 float64 numbers: %f" % sys.getsizeof(cf64))

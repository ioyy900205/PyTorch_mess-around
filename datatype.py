'''
Date: 2021-05-17 20:18:34
LastEditors: Liuliang
LastEditTime: 2021-05-18 09:42:16
Description: 
'''

import numpy as np
import sys
import torch
# 32位整型
ai32 = np.array([], dtype=np.int32)
bi32 = np.arange(1, dtype=np.int32)
ci32 = np.arange(5, dtype=np.int32)

# 64位整型
ai64 = np.array([], dtype=np.int64)
bi64 = np.arange(1, dtype=np.int64)
ci64 = np.arange(5, dtype=np.int64)

# 32位浮点数
af32 = np.array([], dtype=np.float32)
bf32 = np.arange(1, dtype=np.float32)
cf32 = np.arange(5, dtype=np.float32)

# 64位浮点数
af64 = np.array([], dtype=np.float64)
bf64 = np.arange(1, dtype=np.float64)
cf64 = np.arange(5, dtype=np.float64)

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

'''
Date: 2021-06-22 16:15:27
LastEditors: Liuliang
LastEditTime: 2021-06-22 16:17:10
Description: 
'''
import numpy as np


# .npy文件是numpy专用的二进制文件
arr = np.array([[1, 2], [3, 4]])

# 保存.npy文件
np.save("arr", arr)
print("save .npy done")

# # 读取.npy文件
# np.load("../data/arr.npy")
# print(arr)
# print("load .npy done")

# import numpy as np  
# a = np.load("speech-linear-13100.npy") 
# print(a)  
# print("数据类型",type(a))           #打印数组数据类型

# >>> type(a)

# <class 'numpy.ndarray'>

#  
# print("数组元素数据类型：",a.dtype) #打印数组元素数据类型

#  >>> a.dtype

# dtype('float32') 


# print("数组元素总数：",a.size)      #打印数组尺寸，即数组元素总数  

# >>> a.size

# 394497

# print("数组形状：",a.shape)         #打印数组形状

# >>> a.shape

# (769, 513)

#  
# print("数组的维度数目",a.ndim)      #打印数组的维度数目  

# a.ndim

# 2
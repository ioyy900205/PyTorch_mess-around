'''
Date: 2021-08-09 16:43:36
LastEditors: Liuliang
LastEditTime: 2021-08-10 16:24:05
Description: 
'''
import random
import sys 
sys.path.append("..") 
from bacic_module.random_int_list import random_int_list

c = random_int_list(0,10,10)
print(c)


def iss(nums):
    n = len(nums)
    for i in range(1,n):
        while i > 0 and nums[i-1] > nums[i]:
            nums[i-1],nums[i] = nums[i], nums[i-1]
            i -= 1
    return nums

d = iss(c)
print(d)
            
'''
Date: 2021-08-10 16:29:27
LastEditors: Liuliang
LastEditTime: 2021-08-10 16:48:24
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
    gap = n // 2
    while gap:
        for i in range(gap,n):
            while i - gap >=0  and nums[i-gap] > nums[i]:
                nums[i-gap],nums[i] = nums[i], nums[i-gap]
                i -= gap
        gap //= 2
    return nums

d = iss(c)
print(d)
            
'''
Date: 2021-08-09 16:43:36
LastEditors: Liuliang
LastEditTime: 2021-08-09 18:25:15
Description: 
'''
import random
import sys 
sys.path.append("..") 
from bacic_module.random_int_list import random_int_list

c = random_int_list(0,10,10)
print(c)


def sss(nums):
    n = len(nums)
    for i in range(n): 
        for j in range(i,n):
            if nums[i] > nums[j]:
                nums[i], nums[j] = nums[j], nums[i]
        print(nums)
    return nums

d = sss(c)
# print(d)
            
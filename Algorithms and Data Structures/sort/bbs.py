'''
Date: 2021-08-09 16:43:36
LastEditors: Liuliang
LastEditTime: 2021-08-09 17:05:36
Description: 
'''
import random
import sys 
sys.path.append("..") 
from bacic_module.random_int_list import random_int_list

c = random_int_list(0,10,5)
print(c)


def bbs(nums):
    n = len(nums)
    for i in range(n):
        for j in range(1,n-i):
            if nums[j-1] > nums[j]:
                nums[j-1], nums[j] = nums[j], nums[j-1]
        print(nums)
    return nums

d = bbs(c)
# print(d)
            
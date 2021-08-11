'''
Date: 2021-08-10 16:48:48
LastEditors: Liuliang
LastEditTime: 2021-08-10 17:13:13
Description: 
'''

import random
import sys 
sys.path.append("..") 
from bacic_module.random_int_list import random_int_list

c = random_int_list(0,10,10)
def merge(left,right):
    res = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i])
            i += 1
        elif left[i] > right[j]:
            res.append(right[j])
            j += 1
    res += (left[i:]  +  right[j:])
    return res

# c = merge([1,3,5], [2,4,6])
# print(c)

def mg_s(nums):
    if len(nums)<=1: return nums
    mid = len(nums) // 2
    left = mg_s(nums[:mid])
    right = mg_s(nums[mid:])
    return merge(left, right)
c = random_int_list(0,10,10)
d = mg_s(c)
print(d)
        
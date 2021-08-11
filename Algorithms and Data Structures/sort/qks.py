'''
Date: 2021-08-10 17:17:35
LastEditors: Liuliang
LastEditTime: 2021-08-10 18:27:56
Description: 
'''
import random
import sys 
sys.path.append("..") 
from bacic_module.random_int_list import random_int_list

def partition(nums, left, right):    
    tmp = nums[left]
    while left < right:
        while left<right and nums[right] >= tmp:
            right -= 1
        nums[left] = nums[right]
        while left<right and nums[left] <= tmp:
            left += 1
        nums[right] = nums[left]
    nums[left] = tmp
    return left

def qks(nums, left, right):
    if left < right:
        mid = partition(nums,left,right)
        qks(nums,left,mid-1)
        qks(nums,mid+1,right)

    
c = random_int_list(0,10,10)
print(c)

p = qks(c,0,len(c)-1)
print(c)

'''
Date: 2021-08-09 17:05:58
LastEditors: Liuliang
LastEditTime: 2021-08-09 17:06:27
Description: 
'''
import random
import sys 
sys.path.append("..") 
from bacic_module.random_int_list import random_int_list

c = random_int_list(0,10,5)
print(c)

def iss(nums):
    n = len(nums)
    
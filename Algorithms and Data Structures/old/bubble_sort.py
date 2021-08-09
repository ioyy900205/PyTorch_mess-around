'''
Date: 2021-06-22 09:59:40
LastEditors: Liuliang
LastEditTime: 2021-06-22 10:24:49
Description: bubble_sort
'''

import random
# from random_int_list import *
from bacic_module.random_int_list import *
import numpy as np


def bubble_sort(li):
    lenth = len(li)
    for i in range(lenth-1):
        for j in range(lenth-1-i):
            if li[j] > li[j+1]: li[j],li[j+1] = li[j+1], li[j]


if __name__ == "__main__":

    list = random_int_list(5,1000,20)    
    print(list)
    bubble_sort(list)
    print(list)


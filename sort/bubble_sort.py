'''
Date: 2021-06-21 10:50:13
LastEditors: Liuliang
LastEditTime: 2021-06-21 11:51:23
Description: bubble
'''

import random
from cal_time import *


c = [ i for i in range(10)]
random.shuffle(c)

print(c)

def bubble(list):
    lenth = len(list)
    for i in range(lenth-1):
        flag = False
        for j in range(lenth-1-i):
            if c[j] > c[j+1]:
                c[j], c[j+1] = c[j+1], c[j]
                flag = True
        if flag == False:
            return

    
def partition():


    return

bubble(c)
print(c)
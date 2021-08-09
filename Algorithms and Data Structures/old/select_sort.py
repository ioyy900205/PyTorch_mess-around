'''
Date: 2021-06-22 14:17:31
LastEditors: Liuliang
LastEditTime: 2021-06-22 14:31:21
Description: select_sort
'''
from random import seed
from bacic_module.random_int_list import random_int_list
import random
random.seed(0)

def select_sort(li):
    lenth = len(li)
    c = []
    for i in range(lenth):        
        c.append(min(li))
        li.pop(li.index(min(li)))
    for i in c:
        li.append(i)
    del c
    
    
    


if __name__ == "__main__":

    c = random_int_list(5,1000,20)
    print(c)
    select_sort(c)
    print(c)


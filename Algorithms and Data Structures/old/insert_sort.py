'''
Date: 2021-06-22 10:25:01
LastEditors: Liuliang
LastEditTime: 2021-08-09 16:40:55
Description: insert
'''

from random import seed
from bacic_module.random_int_list import random_int_list
import random
random.seed(0)


def insert_sort(li):
    lenth = len(li)
    for i in range(1,lenth): 
        j = i-1
        tmp = li[i]            
        while j>=0 and li[j]>tmp:
            
            li[j+1] = li[j]
            j -= 1
        li[j+1] = tmp
    



if __name__ == "__main__":
    
    print('insert_sort test')
    c = random_int_list(5,1000,20)
    print(c)
    insert_sort(c)
    print(c)
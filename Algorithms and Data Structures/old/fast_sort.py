'''
Date: 2021-06-22 14:31:45
LastEditors: Liuliang
LastEditTime: 2021-06-22 15:38:08
Description: fast_sort
'''
from random import seed
from bacic_module.random_int_list import random_int_list
import random
random.seed(0)

def partition(li,left,right):

    # left = 0
    # right = lenth-1
    tmp = li[left]
    while left < right:
        while left < right and li[right]>tmp:
            right -= 1
        li[left] = li[right]
        while left < right and li[left]<tmp:
            left += 1
        li[right] = li[left]
    li[left] = tmp
    return left



def fast_sort(li,left,right):
    if left<right:
        mid = partition(li,left,right)
        fast_sort(li,left,mid-1)
        fast_sort(li,mid+1,right)       




if __name__ == "__main__":
    
    print('quick_sort test')
    c = random_int_list(0,1000,40)
    # c = [5,7,4,6,3,1,2,9,8]
    print(c)
    fast_sort(c,0,len(c)-1)
    print(c)
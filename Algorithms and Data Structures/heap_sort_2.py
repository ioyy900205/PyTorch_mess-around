'''
Date: 2021-06-28 10:22:11
LastEditors: Liuliang
LastEditTime: 2021-06-28 10:29:58
Description: 
'''

def sift(list,low,high):
    i = low
    j = 2 * i + 1
    tmp = list[low]
    while j <= high:
        if j + 1 <= high and list[j] > list[j+1]:
            j = j + 1
        if tmp > list[j]:
            list[i] = list[j]
            i = j
            j = 2 * i + 1
        else:
            break
    else:
        list[i] = tmp

def heap_sort(li):
    n = len(li)
    #build heap
    for i in range((n-2//2),-1,-1):
        sift(li,i,n-1)
    #sort heap
    for i in range(n-1,-1,-1):
        li[0],li[i] = li[i], li[0]
        sift(li, 0, i-1)

import random
list = [x for x in range(10)]
random.shuffle(list)
print(list)
sift(list,0,9)
print(list)
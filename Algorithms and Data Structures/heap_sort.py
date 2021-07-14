'''
Date: 2021-06-22 17:16:25
LastEditors: Liuliang
LastEditTime: 2021-07-05 17:11:39
Description: heap
'''

def sift(li, low, high):
    """heap_sort

    Args:
        li (list): the list for process
        low (top position): the top of the heap
        high (last postion): the last position of the heap
    """
    i = low
    j = 2 * i + 1
    tmp = li[low]
    while j <= high :
        if j + 1 <= high and li[j] > li[j+1]: j = j + 1 #shfit the position of j
        if tmp > li[j]:
            li[i] = li[j]
            i = j
            j = 2 * i + 1
        else:
            break
    else:
        li[i] = tmp
    
def heap_sort(li):
    n = len(li)
    # build the heap
    for i in range((n-2)//2, -1, -1):
        sift(li, i, n-1)
    # sort heap
    for i in range(n-1, -1, -1):
        li[0] ,li[i] = li[i], li[0]
        sift(li, 0, i-1)
        
def topk(li,k):
    heap = li[0:k]
    #build the heap
    for i in range((k-2)//2, -1, -1):
        sift(heap, i, k-1)
    
    for i in range(k,len(li)-1):
        if li[i] > heap[0]:
            heap[0] = li[i]
        sift(heap,0,k-1)
    # output
    for i in range(k-1,-1,-1):
        heap[0],heap[i] = heap[i], heap[0]
        sift(heap, 0, i-1)
    print(heap)



list = [x for x in range(10)]
import random
random.shuffle(list)
print(list)
# heap_sort(list)
# print(list)
topk(list,5)

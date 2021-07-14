'''
Date: 2021-07-13 11:17:34
LastEditors: Liuliang
LastEditTime: 2021-07-13 11:56:54
Description: binarySearch
'''
def find_binary(list,num):
    left = 0
    right = len(list)-1
    while left < right:
        mid = left + (right - left) // 2
        if list[mid] < num: left = mid + 1
        else: right = mid
    return left

if __name__ == '__main__':
    list = [2,2,2,5,6]
    c = find_binary(list,2)
    print(c)
    c = find_binary(list,3)
    print(c)
    c = find_binary(list,4)
    print(c)
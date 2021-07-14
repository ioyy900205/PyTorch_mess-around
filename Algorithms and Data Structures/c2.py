'''
Date: 2021-07-13 15:13:49
LastEditors: Liuliang
LastEditTime: 2021-07-13 15:40:21
Description: 
'''
class solution:
    def search_left(self,nums,target):
        left = 0
        right = len(nums)
        if right == 0:
            return 0
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left
    
    def search_right(self,nums,target):
        
        left = 0
        right = len(nums)
        if right == 0:
            return 0
        
        while left < right:
            mid = (left + right - 1) // 2
            if nums[mid] > target:
                right = mid - 1
            else:
                left = mid
        return right 
 

c = solution()
list = [1,3,5,6]
m = c.search_right(nums = list,target = 6)
print(m)



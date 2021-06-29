'''
Date: 2021-06-28 17:29:50
LastEditors: Liuliang
LastEditTime: 2021-06-28 17:37:52
Description: lis problem
'''

class Solution:
    def LIS(self,nums):
        #creat the dp table
        dp = [1 for i in range(len(nums))]
        m = 0
        for i in range(1,len(nums)):
            for j in range(0,len(nums)):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j]+1)
            m = max(dp[i], m)
         
        return m


s = Solution()
list = [5,4,1,2,3,3,2,3,23,23,2,3,3]
print(s.LIS(list))

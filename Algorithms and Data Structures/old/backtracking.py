'''
Date: 2021-07-12 10:48:27
LastEditors: Liuliang
LastEditTime: 2021-07-12 11:33:16
Description: 
'''

def permute(nums):

    if len(nums) == 0:
        return []
    res = []
    nums.sort()

    def _backtracing(nums,pre_list):
        if len(nums)==0:
            res.append(pre_list.copy())
            return
        else:
            for i in range(len(nums)):
                if i>0 and nums[i] == nums[i-1]: continue
                
                pre_list.append(nums[i])
                num_cpy = nums.copy()
                num_cpy.remove(nums[i])
                _backtracing(num_cpy, pre_list)
                pre_list.pop()
        return
        
    _backtracing(nums,[])
    return res



c = permute([1,1,3])

print(c)



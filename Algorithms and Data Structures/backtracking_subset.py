'''
Date: 2021-07-12 11:37:24
LastEditors: Liuliang
LastEditTime: 2021-07-13 11:14:48
Description: 
'''

def subsets(nums,target):
    res = []
    if len(nums) == 0: 
        return res
    nums.sort()

    def _backtracing(nums,pre_list):
        if sum(pre_list) == target: 
            res.append(pre_list.copy())
        for i in range(len(nums)):
            left_num = target - nums[i] - sum(pre_list)
            if left_num < 0: break
            if i>0 and nums[i] == nums[i-1]:
                continue
            pre_list.append(nums[i])   
            _backtracing(nums[i+1:],pre_list)
            pre_list.pop()
            
    _backtracing(nums,[])

    return res




if __name__ == '__main__':
    nums =   [10,1,2,7,6,1,5]
    target = 8
    c = subsets(nums,target)
    print(c)
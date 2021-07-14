'''
Date: 2021-07-06 16:29:40
LastEditors: Liuliang
LastEditTime: 2021-07-06 16:40:19
Description: 
'''

list_1 = [1,1,2,2,3,3,4,4]

c = set(list_1)

print(c)
dic = {}
for i in c:
    dic[i] = 0
    
for j in list_1:
    dic[j] += 1

print(dic)

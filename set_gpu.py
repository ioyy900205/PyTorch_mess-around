'''
Date: 2021-05-28 10:38:31
LastEditors: Liuliang
LastEditTime: 2021-05-28 11:00:31
Description: 设置GPU
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch



# c = torch.randn(7)
# c.cuda()
# print(c.device)


# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
c = torch.randn(7).cuda()
# c = c.to(device)
print(c)
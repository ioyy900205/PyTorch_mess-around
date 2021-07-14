'''
Date: 2021-07-03 17:36:14
LastEditors: Liuliang
LastEditTime: 2021-07-05 17:18:58
Description: BCE_logic
'''

import torch
import torch.nn as nn
from torch.random import seed
import random

SEED = 0
torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# random.seed(SEED)
# c =  random.randint(0,9)

# print(c)


input = torch.rand(3,3)
sig = nn.Sigmoid()

c = sig(input)  

# print(input)
print(c)

target = torch.FloatTensor([
    [0,1,1],
    [0,0,1],
    [1,0,1]
])

# print(target)
# loss = nn.BCELoss()
# val_loss = loss(c,target)

# print(val_loss)

# #验证BCEloss
loss_2 = nn.BCEWithLogitsLoss()
val_loss_2 = loss_2(input,target)
print(val_loss_2)
'''
Date: 2021-07-06 16:47:40
LastEditors: Liuliang
LastEditTime: 2021-07-06 17:10:43
Description: 
'''
import numpy as np
import pandas as pd
import torch 
from torch import nn 
import torch.nn.functional as F
from torch.nn.modules.loss import L1Loss 

# ================================================================== #
#                说明：L1Loss                                             
# ================================================================== #	
x_i = torch.tensor([15.,10.])
y_i = torch.tensor([20.,20.])


l1 = nn.L1Loss()
print(l1(x_i,y_i))

l1__=torch.sum((y_i-x_i))/len(x_i)
print(l1__)

# ================================================================== #
#                说明：MSELoss                                             
# ================================================================== #	
l2 = nn.MSELoss()
print(l2(x_i,y_i))

l2__= torch.sum((y_i-x_i)**2)/len(x_i)
print(l2__)

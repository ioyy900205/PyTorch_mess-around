'''
Date: 2021-05-20 10:45:33
LastEditors: Liuliang
LastEditTime: 2021-05-20 10:50:54
Description: bottleneck
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = 1       
        return out


y = Bottleneck(in_planes=1,growth_rate=1)

print(y)

x = y(1)

print(x)
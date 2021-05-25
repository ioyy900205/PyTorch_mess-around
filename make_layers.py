'''
Date: 2021-05-21 10:06:37
LastEditors: Liuliang
LastEditTime: 2021-05-21 12:59:25
Description: 测试make_layers函数的作用,从densenet中摘取出来
'''

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
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out

        
nblocks = [6,12,24,16]
block1 = Bottleneck
num_planes = 12

def _make_dense_layers(block, in_planes, nblock):
    layers = []
    growth_rate = 12
    for i in range(nblock):
        layers.append(block(in_planes, growth_rate))
        in_planes += growth_rate
    return nn.Sequential(*layers)


layer = _make_dense_layers(
    block=block1,           #基本块
    in_planes=num_planes,   #输入的通道数
    nblock=nblocks[0]       #生产块的个数
    )

print(layer)


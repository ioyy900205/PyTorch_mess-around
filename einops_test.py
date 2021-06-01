'''
Date: 2021-06-01 14:44:59
LastEditors: Liuliang
LastEditTime: 2021-06-01 18:05:33
Description: einops test
'''
import einops
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import os
from einops import rearrange


images = np.array(Image.open('./flower_test.jpg'))

# print(images.shape)


# plt.title('1')
# plt.imshow(images)
# plt.show

x = rearrange(images,'h w c -> h w c')
# print(x.shape)

x = rearrange(x, '(h p1) (w p2) c -> (h w) p1 p2 c', p1=30, p2=30)

# print(x.shape)
# for i in range(100):
    # plt.title()
plt.imshow(x[48])
plt.show

#注意，这里bchw仅仅只是符号，只代表维度顺序，例如numpy中，第四个为c 所以这里第四位是c
#而pytorch中，第二位是c
import torch
c = torch.randn(1, 64 + 1, 1024)
print(c.size())
# print(c)

b = c[:, :]
print(b.size())


a = torch.Tensor([[1,2,4]])
b = torch.Tensor([[4,5,7], [3,9,8], [9,6,7]])
c = torch.cat((a,b), dim=0)
print(c)
print(c.size())
print('********************')
d = torch.chunk(c,2,dim=1)
print(d)
print(len(d))
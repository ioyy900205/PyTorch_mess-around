'''
Date: 2021-06-22 16:05:24
LastEditors: Liuliang
LastEditTime: 2021-06-22 16:14:16
Description: 
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

image = np.load("/media/data/data02/nyuv2/depths/2.npy")
image_pl = np.expand_dims(image,axis=2)
plt.imshow(image_pl)
# plt.imshow(image_pl, cmap ='gray')
plt.show()

'''
Date: 2021-06-08 19:49:40
LastEditors: Liuliang
LastEditTime: 2021-06-08 19:55:16
Description: enforce
'''
import numpy as np # linear algebra
import cv2
import matplotlib.pyplot as plt
import albumentations as albu

image = cv2.imread('/home/liuliang/leaves/images/0.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
aug = albu.HorizontalFlip(p=1)
img_HorizontalFlip = aug(image=image)
# image = cv2.imread('../input/neuralstyletransfersample-photo/Yangshiyi.png')
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# aug = albu.HorizontalFlip(p=1)
# img_HorizontalFlip = aug(image=image)['image']
aug = albu.ShiftScaleRotate(p=1,shift_limit=0.05,rotate_limit = 45)
img_ShiftScaleRotate = aug(image=image)['image']

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(image)
plt.subplot(1,3,2)
plt.imshow(img_HorizontalFlip)
plt.subplot(1,3,3)
plt.imshow(img_ShiftScaleRotate)
'''
Date: 2021-05-11 12:02:31
LastEditors: Liuliang
LastEditTime: 2021-05-19 09:49:56
Description: AnchorGenerator
'''


import torchvision.models.detection.rpn as rpn
import torchvision.models.detection.image_list as image_list
import torch
 
# 创建AnchorGenerator实例
anchor_generator = rpn.AnchorGenerator()
 
# 构建ImageList
batched_images = torch.Tensor(8,3,640,640)
image_sizes = [(640,640)] * 8
image_list_ = image_list.ImageList(batched_images,image_sizes)
 
# 构建feature_maps
feature_maps = [torch.Tensor(8,256,80,80),torch.Tensor(8,256,160,160), torch.Tensor(8,256,320,320)]
 
# 生成anchors
anchors = anchor_generator(image_list_,feature_maps)

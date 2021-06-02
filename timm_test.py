'''
Date: 2021-06-02 10:58:25
LastEditors: Liuliang
LastEditTime: 2021-06-02 12:20:36
Description: timm 使用      
'''
import timm
from pprint import pprint
model_names = timm.list_models('*efficientnet_l2*')
pprint(model_names)
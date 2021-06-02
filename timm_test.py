'''
Date: 2021-06-02 10:58:25
LastEditors: Liuliang
LastEditTime: 2021-06-02 11:00:45
Description: timm 使用      
'''
import timm

import timm
from pprint import pprint
model_names = timm.list_models(pretrained=True)
pprint(model_names)
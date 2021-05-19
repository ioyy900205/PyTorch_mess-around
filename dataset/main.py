'''
Date: 2021-05-11 14:18:25
LastEditors: Liuliang
LastEditTime: 2021-05-11 14:40:48
Description: 主要的函数 调用transforms；VOC2012DataSet
'''

import transforms
from draw_box_utils import draw_box
from VOC2012DataSet import VOC2012DataSet
from PIL import Image
import json
import matplotlib.pyplot as plt
import torchvision.transforms as ts
import random

# read class_indict
category_index = {}
try:
    json_file = open('./pascal_voc_classes.json', 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)

data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(0.5)]),
    "val": transforms.Compose([transforms.ToTensor()])
}

# load train data set
ospath = "/home/liuliang/dataset_python/"
train_data_set = VOC2012DataSet(ospath, data_transform["train"], "train.txt")
print("len(train_data_set):",len(train_data_set))
for index in random.sample(range(0, len(train_data_set)), k=1):
    # img, target = train_data_set[index]
    img, target = train_data_set[5292]
    # print(index)
    # img, target = train_data_set[2]
    img = ts.ToPILImage()(img)
    # plt.imshow(img)
    # plt.show()
    # input()
    draw_box(img,
             target["boxes"].numpy(),
             target["labels"].numpy(),
             [1 for i in range(len(target["labels"].numpy()))],
             category_index,
             thresh=0.5,
             line_thickness=5)
    plt.imshow(img)
    plt.show()


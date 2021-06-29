'''
Date: 2021-05-20 11:12:16
LastEditors: Liuliang
LastEditTime: 2021-05-20 16:56:33
Description: tqdm test
'''

from tqdm import tqdm
import torch
from my_dataset import CocoDetection
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from time import sleep as sleep
import time
import os 

# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                     transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
# COCO_root = '/media/data/data02/COCO2017'

# train_data_set = CocoDetection(COCO_root, "train", data_transform["train"])

# train_loader = torch.utils.data.DataLoader(train_data_set,
#                                                     batch_size=32,
#                                                     shuffle=True,
#                                                     pin_memory=True,
#                                                     # num_workers=8,
#                                                     collate_fn=train_data_set.collate_fn)
# # i = 0
# imgs = 0
# target = 0
# trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# train_data_loader = torchvision.datasets.MNIST(
#     root= "/home/liuliang/dataset_python/",
#     train=True,
#     transform=trans,
#     download=True) 

# batch_size = 300

traindir = os.path.join('/media/data/data02/Imagenet2012/', 'train')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    )

train_loader = torch.utils.data.DataLoader(
                 dataset=train_dataset,
                 batch_size=128,
                 num_workers=16,
                 shuffle=False)
                 
a1 = time.time()
for batch_idx, (x, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
    # print(batch_idx,x.size(),target.size())
    # input()
    pass
a2 = time.time()
print(a2-a1)


a1 = time.time()
for batch_idx, (x, target) in enumerate(train_loader):
    # print(batch_idx,x.size(),target.size())
    # input()
    pass
a2 = time.time()
print(a2-a1)

# count = 0
# for batch_idx, item in tqdm(enumerate(train_loader)):
#     count+=batch_idx
# print(count)


# for data, target in tqdm(train_data_loader):
#     print(data)
#     pass

#用list方式.

# for i in tqdm(range(10000)):  
#      #do something
#      sleep(0.01)
#      pass

# for char in tqdm(["a", "b", "c", "d"]):
#     #do something
#     pass

# pbar = tqdm(["a", "b", "c", "d"])
# for char in pbar:
#     sleep(1)
#     pbar.set_description("Processing %s" % char)
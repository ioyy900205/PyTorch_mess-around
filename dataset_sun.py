'''
Date: 2021-06-18 18:41:27
LastEditors: Liuliang
LastEditTime: 2021-06-19 10:19:21
Description: 处理数据
'''

#coding=utf-8
import os
import numpy as np
'''训练集'''
path_train_img ='/media/data/data02/SUNRGB-D/train/rgb/'
path_train_depth ='/media/data/data02/SUNRGB-D/train/depth/'
path_train_label ='/media/data/data02/SUNRGB-D/train/label/'

'''验证集'''
path_test_img = '/media/data/data02/SUNRGB-D/test/rgb/'
path_test_depth = '/media/data/data02/SUNRGB-D/test/depth/'
path_test_label = '/media/data/data02/SUNRGB-D/test/label/'


data_list=[]
data_list_2 = []

def data_process(path,folder_name):
    fns = [os.path.join(root,fn) for root, dirs, files in os.walk(path) for fn in files]
    c = []
    for f in fns:
        c.append(f+'\n')
    return c

train_list = [path_train_img,path_train_depth,path_train_label]
test_list  = [path_test_img,path_test_depth,path_test_label]

for path in train_list:
    path_spilt = list(filter(None, path.rstrip().split('/')))
    folder_name = path_spilt[-2]+'_'+path_spilt[-1]
    data = data_process(path,folder_name)
    data_list.append(data)

txt_1 = open("train_SUN_RGBD.txt",'w')

for i in range(len(data_list[0])):
    c = data_list[0][i].replace('\n', '')+ ' ' +data_list[1][i].replace('\n', '')+ ' '+data_list[2][i].replace('\n', '')
    txt_1.writelines(c)  
    txt_1.write('\n')

for path in test_list:
    path_spilt = list(filter(None, path.rstrip().split('/')))
    folder_name_2 = path_spilt[-2]+'_'+path_spilt[-1]
    data_2 = data_process(path,folder_name_2)
    data_list_2.append(data_2)
print(data_list_2[0])


txt_2 = open("test_SUN_RGBD.txt",'w')

for i in range(len(data_list_2[0])):
    c = data_list_2[0][i].replace('\n', '')+ ' ' +data_list_2[1][i].replace('\n', '')+ ' '+data_list_2[2][i].replace('\n', '')
    txt_2.writelines(c)  
    txt_2.write('\n')


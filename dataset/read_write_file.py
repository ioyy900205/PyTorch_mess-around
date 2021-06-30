'''
Date: 2021-06-30 14:11:35
LastEditors: Liuliang
LastEditTime: 2021-06-30 14:30:00
Description: read_file
'''
# path 可以自己设置
path = '/home/liuliang/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/data/my_train_data.txt'
path_2 = './test.txt'

def read_file(path):
    with open(path,'r') as f:
        f = f.read().splitlines()
    return f

c = read_file(path)
#write_file
with open(path_2,'w') as f:
    for i in c:
        str =i.split('/')[-1]+'\n'
        f.writelines(str)
        
    
    





'''
Date: 2021-06-08 10:04:21
LastEditors: Liuliang
LastEditTime: 2021-06-29 09:16:55
Description: main
'''

# 首先导入包
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.transforms.transforms import RandomCrop
# This is for the progress bar.
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import ipdb
import albumentations
import cv2
import ttach as tta

#CutMix方法
"""输入为：样本的size和生成的随机lamda值"""
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    """1.论文里的公式2，求出B的rw,rh"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
 
    # uniform
    """2.论文里的公式2，求出B的rx,ry（bbox的中心点）"""
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    #限制坐标区域不超过样本大小
 
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    """3.返回剪裁B区域的坐标值"""
    return bbx1, bby1, bbx2, bby2



os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
torch.backends.cudnn.benchmark = True
print(device)

tb_writer = SummaryWriter(log_dir="runs/flower_experiment")

# 看看label文件长啥样
labels_dataframe = pd.read_csv('/home/liuliang/leaves/train.csv')


# 把label文件排个序
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
class_to_num = dict(zip(leaves_labels, range(n_classes)))
num_to_class = {v : k for k, v in class_to_num.items()}

class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """
        
        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path)  #header=None是去掉表头部分
        self.data_info = self.data_info.sample(frac=1.0)
        # 计算 length
        self.data_len = len(self.data_info.index)
        self.train_len = int(self.data_len * (1 - valid_ratio))
        
        if mode == 'train':
            # 第一列包含图像文件的名称
            # self.train_image = np.asarray(self.data_info.iloc[1:, 0])
            self.train_image = np.asarray(self.data_info.iloc[0:self.train_len, 0])  #self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            # 第二列是图像的 label
            # self.train_label = np.asarray(self.data_info.iloc[1:, 1])
            self.train_label = np.asarray(self.data_info.iloc[0:self.train_len, 1])

            self.image_arr = self.train_image 
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])  
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image
            
        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)

        #如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
#         if img_as_img.mode != 'L':
#             img_as_img = img_as_img.convert('L')

        #设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                # transforms.CenterCrop(224),
                # transforms.TenCrop(224, vertical_flip=False),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(size=224),
                # transforms.TenCrop
                # transforms.RandomCrop(224),
                # transforms.RandomVerticalFlip() ,
                # transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                # transforms.RandomRotation(degrees=15),
                # transforms.RandomHorizontalFlip(),
                #transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])



        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        
        img_as_img = transform(img_as_img)
        
        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]

            return img_as_img, number_label  #返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len

    

train_path = '/home/liuliang/leaves/train.csv'
test_path = '/home/liuliang/leaves/test.csv'
# csv文件中已经images的路径了，因此这里只到上一级目录
img_path = '/home/liuliang/leaves/'

train_dataset = LeavesData(train_path, img_path, mode='train')
val_dataset = LeavesData(train_path, img_path, mode='valid')
test_dataset = LeavesData(test_path, img_path, mode='test')
batch_size = 64
# 定义data loader
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
 
# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False




# resnext101_32x8d
def res_model(num_classes, feature_extract = False, use_pretrained=True):

    model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

# 超参数
learning_rate = 3e-4
weight_decay = 1e-3
num_epoch = 50
beta = 0.9
cutmix_prob = 0.9
model_path = './pre_res_model2.ckpt'

# torch.distributed.init_process_group(backend="nccl")
model = res_model(176)
model = nn.DataParallel(model)
model = model.to(device)
# model = nn.parallel.DistributedDataParallel(model) # device_ids will include all GPU devices by default


# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=0.02)
# optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-6, last_epoch=-1)
# The number of training epochs.
n_epochs = num_epoch

# ipdb.set_trace()
best_acc = 0.0
for epoch in range(n_epochs):
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train() 
    # These are used to record information in training.
    train_loss = []
    train_accs = []
    # Iterate the training set by batches.
    for batch in tqdm(train_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        r = np.random.rand(1)

        if beta > 0 and r < cutmix_prob:
            # generate mixed sample
            """1.设定lamda的值，服从beta分布"""
            lam = np.random.beta(beta, beta)
            """2.找到两个随机样本"""
            rand_index = torch.randperm(imgs.size()[0]).cuda()
            target_a = labels#一个batch
            target_b = labels[rand_index] #batch中的某一张
            """3.生成剪裁区域B"""
            bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
            """4.将原有的样本A中的B区域，替换成样本B中的B区域"""
            imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            """5.根据剪裁区域坐标框的值调整lam的值"""
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
            # compute output
            """6.将生成的新的训练样本丢到模型中进行训练"""
            logits = model(imgs)
            """7.按lamda值分配权重"""
            loss = criterion(logits, target_a) * lam + criterion(logits, target_b) * (1. - lam)
        else:
                # compute output
            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs)
            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels)

        
        
        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()
        # Compute the gradients for parameters.
        loss.backward()
        # Update the parameters with computed gradients.
        optimizer.step()
        
        
        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
        
    scheduler.step()    
    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f},best_acc = {best_acc:.5f}")
    
    
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()
    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []
    
    # Iterate the validation set by batches.
    for batch in tqdm(val_loader):
        imgs, labels = batch
        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))
            
        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        
    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}, lr = {optimizer.param_groups[0]['lr']:.5f}")

    tb_writer.add_scalar("train_loss", train_loss, epoch)
    tb_writer.add_scalar("train_acc", train_acc, epoch)
    tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)
    tb_writer.add_scalar("valid_loss",valid_loss,epoch)
    tb_writer.add_scalar("valid_acc",valid_acc,epoch)
  
  

    
    # if the model improves, save a checkpoint at this epoch
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), model_path)
        print('saving model with acc {:.3f}'.format(best_acc))

tb_writer.close()  

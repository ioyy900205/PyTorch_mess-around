import torch
from torch import nn, optim
from torchvision import transforms
import timm

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import sys
import codecs

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.backends.cudnn.benchmark = True
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

train_csv = pd.read_csv('/home/liuliang/leaves/train.csv')

leaves_labels = sorted(list(set(train_csv['label'])))
n_classes = len(leaves_labels)
class_to_num = dict(zip(leaves_labels, range(n_classes)))
num_to_class = {v : k for k, v in class_to_num.items()}





class ReadData(torch.utils.data.Dataset):
    def __init__(self, csv_data, transform=None):
        super(ReadData, self).__init__()
        self.data = csv_data
        self.transform = transform
        
    def __getitem__(self, idx): #idx，找出对应的图像和标签
        img = Image.open("/home/liuliang/leaves/" + self.data.loc[idx, "image"])
        label = class_to_num[self.data.loc[idx, "label"]]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
        
    def __len__(self):
        return len(self.data)
    
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def kfold(data, k=5):
    """ K折交叉验证 """
    
    KF = KFold(n_splits=k, shuffle=False)
    for train_idxs, test_idxs in KF.split(data):
        train_data = data.loc[train_idxs].reset_index(drop=True)
        valid_data = data.loc[test_idxs].reset_index(drop=True)
        train_iter = torch.utils.data.DataLoader(
            ReadData(train_data, train_transform), batch_size=64,
            shuffle=True, num_workers=8, pin_memory=True
        )

        valid_iter = torch.utils.data.DataLoader(
            ReadData(valid_data, valid_transform), batch_size=64,
            shuffle=True, num_workers=8, pin_memory=True
        )
        
        yield train_iter, valid_iter

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """ Mixup 数据增强 -> 随机叠加两张图像 """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    """ 随机裁剪 """
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    """ Cutmix 数据增强 -> 随机对主图像进行裁剪, 加上噪点图像
    W: 添加裁剪图像宽
    H: 添加裁剪图像高
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def get_models(k=5):
    models = {}
    for mk in range(k):
        model = timm.create_model("resnest50d_4s2x40d", True, drop_rate=.5)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(.3),
            nn.Linear(512, len(num_to_class))
        )
        # for param in model.layer4.parameters():
        #     if isinstance(param, nn.Conv2d):
        #         nn.init.xavier_normal_(param.weight)
        # for param in model.fc.parameters():
        #     if isinstance(param, nn.Linear):
        #         nn.init.kaiming_normal_(param.weight)
        # model.load_state_dict(torch.load(f"../input/resnest50/Resnest50d.pth"))
        for i, param in enumerate(model.children()):
            if i == 6:
                break
            param.requires_grad = False

        model.cuda()

        opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 10, T_mult=2)
        models[f"model_{mk}"] = {
            "model": model,
            "opt": opt,
            "scheduler": scheduler,
            "last_acc": .97
        }
    
    return models

models = get_models()


def mixup_criterion(pred, y_a, y_b, lam):
    c = nn.CrossEntropyLoss()
    return lam * c(pred, y_a) + (1 - lam) * c(pred, y_b)
criterion = mixup_criterion

def train_model():
    for epoch in range(100):
        flod_train_acc = []
        flod_valid_acc = []
        for k, (train_iter, valid_iter) in enumerate(kfold(train_csv, 5)):
            model = models[f"model_{k}"]["model"]
            model = nn.DataParallel(model)
            opt = models[f"model_{k}"]["opt"]
            scheduler = models[f"model_{k}"]["scheduler"]
            s = time.time()
            model.train()
            train_loss = []
            train_acc = 0
            length = 0
            for x, y in train_iter:
                x, y = x.cuda(), y.cuda()
                random_num = np.random.random()
                if random_num <= 1/3:
                    x, y_a, y_b, lam = mixup_data(x, y, use_cuda=True)
                elif random_num <= 2/3:
                    x, y_a, y_b, lam = cutmix_data(x, y, use_cuda=True)
                else:
                    x, y_a, y_b, lam = mixup_data(x, y, alpha=0, use_cuda=True)
                x, y_a, y_b = map(torch.autograd.Variable, (x, y_a, y_b))
                output = model(x)
                loss = criterion(output, y_a, y_b, lam)
                train_loss.append(loss.item())
                predict = output.argmax(dim=1)
                length += x.shape[0]
                train_acc += lam * (predict == y_a).cpu().sum().item() + \
                            (1 - lam) * (predict == y_b).cpu().sum().item()
                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step()

            model.eval()
            valid_acc = []
            with torch.no_grad():
                for x, y in valid_iter:
                    x, y = x.cuda(), y.cuda()
                    pre_x = model(x)
                    valid_acc.append((pre_x.argmax(1) == y).float().mean().item())

            k_train_ = train_acc / length
            k_valid_ = sum(valid_acc) / len(valid_acc)
            if k_valid_ > models[f"model_{k}"]["last_acc"]:
                torch.save(model.state_dict(), f"/home/liuliang/deep_learning/PyTorch_mess-around/kaggle/leaves/Resnest50d_{k}.pth")
                models[f"model_{k}"]["last_acc"] = k_valid_

            response = f"Epoch {epoch + 1}-Fold{k + 1} —— " + \
                    f"Train Loss: {sum(train_loss) / len(train_loss) :.3f}, " + \
                    f"Train Accuracy: {k_train_ * 100 :.2f}%, " + \
                    f"Valid Accuracy: {k_valid_ * 100 :.2f}%, " + \
                    f"Learning Rate: {opt.param_groups[0]['lr'] :.6f}, " + \
                    f"Time Out: {time.time() - s :.1f}s"
            print(response)
            flod_train_acc.append(k_train_)
            flod_valid_acc.append(k_valid_)

        t_accuracy = np.mean(flod_train_acc)
        v_accuracy = np.mean(flod_valid_acc)
        print(f"Epoch {epoch + 1} —— " + \
            f"Train Accuracy: {t_accuracy * 100 :.2f}%, " + \
            f"Valid Accuracy: {v_accuracy * 100 :.2f}%\n")

train_model()
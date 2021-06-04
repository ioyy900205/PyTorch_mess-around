import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import random_split

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd

TRAIN_CSV = "/home/liuliang/deep_learning/PyTorch_mess-around/kaggle/train.csv"
TEST_CSV = "/home/liuliang/deep_learning/PyTorch_mess-around/kaggle/test.csv"
OUT_CSV = "/home/liuliang/deep_learning/PyTorch_mess-around/kaggle/sub-digit-recognizer.csv"

data_read = pd.read_csv(TRAIN_CSV)
data_n = np.array(data_read)
# print(data_read)
print(data_n)
print(data_n.shape)
data_x_t = torch.tensor(data_n[:,1:])
print(data_x_t)
print(data_x_t.shape)
data_x_t = data_x_t.reshape(-1,1,28,28)
print(data_x_t)
print(data_x_t.shape)
data_y_t = torch.tensor(data_n[:,0])
print(data_y_t)
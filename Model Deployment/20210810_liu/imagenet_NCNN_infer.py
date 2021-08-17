'''
Date: 2021-08-10 14:12:17
LastEditors: Liuliang
LastEditTime: 2021-08-17 14:08:50
Description: onnx在imagenet上的推理验证
'''
import argparse
import os
import random
import shutil
import time
import warnings


from Average_ import AverageMeter
from Progress_ import ProgressMeter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import onnxruntime
import numpy as np
import ncnn
from ncnn.model_zoo.model_store import get_model_file


# ================================================================== #
#                说明：设置随机数，保证结果可以复现                                             
# ================================================================== #	
seed = 0
random.seed(seed)
torch.manual_seed(seed)
cudnn.deterministic = True
cudnn.benchmark = True

# ================================================================== #
#                说明：验证集op         batch_size,num_workers可以自己设置                                  
# ================================================================== #	
val_data = '/media/data/data02/Imagenet2012/'
valdir = os.path.join(val_data, 'val')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

# ================================================================== #
#                说明：定义损失                                             
# ================================================================== #	       
criterion = nn.CrossEntropyLoss()

# ================================================================== #
#                说明：定义模型                                             
# ================================================================== #	

class SqueezeNet:
    def __init__(self, target_size=224, num_threads=1, use_gpu=False):
        self.target_size = target_size
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = [104.0, 117.0, 123.0]
        self.norm_vals = []

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        
        self.net.load_param("/home/liuliang/.ncnn/models/resnet50.param")
        self.net.load_model("/home/liuliang/.ncnn/models/resnet50.bin")

    def __del__(self):
        self.net = None

    def __call__(self, img):
        img_h = img.shape[0]
        img_w = img.shape[1]

        # mat_in = ncnn.Mat.from_pixels_resize(
        #     img,
        #     ncnn.Mat.PixelType.PIXEL_BGR,
        #     img.shape[1],
        #     img.shape[0],
        #     self.target_size,
        #     self.target_size,
        # )

        mat_in = ncnn.Mat(img)

        # mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)

        ex.input("input", mat_in)
        
        ret, mat_out = ex.extract("output")
        # print("%d %d %d\n", mat_out.w, mat_out.h, mat_out.c)

        out = np.array(mat_out)
        return out


model = SqueezeNet()

# ================================================================== #
#                说明：def验证                                             
# ================================================================== #	
def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        # compute output
        # output = model(images)
        output = torch.zeros(1,1000)
        for image in images:            
            _output = model(to_numpy(image))
            _output = torch.from_numpy(_output)
            _output = torch.unsqueeze(_output,0)
            output = torch.cat((output, _output),0)            
        output = output[1:]      
            
        # output = output.argmax(axis=1)
        # print(out)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))






def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

validate(val_loader, model, criterion)


import torch
import sys
import numpy as np
device = torch.device('cuda:0')


# 定义tensor


a1 = torch.zeros([2, 4], dtype=torch.float64)

print("size of 4 int32 number: %f" % sys.getsizeof(a1))

dummy_tensor_4 = torch.randn(100, 3, 512, 512).float().to(device) #847    300+547M  #1147
dummy_tensor_5 = torch.randn(100, 3, 512, 512).float().to(device) #1147   300M      #1747
dummy_tensor_6 = torch.randn(100, 3, 512, 512).float().to(device) #1447   300M      #2347


dummy_tensor_4 = dummy_tensor_4.cpu()
dummy_tensor_5 = dummy_tensor_5.cpu()
# 这里虽然将上面的显存释放了，但是我们通过Nvidia-smi命令看到显存依然在占用
torch.cuda.empty_cache()


input()
# 只有执行完上面这句，显存才会在Nvidia-smi中释放
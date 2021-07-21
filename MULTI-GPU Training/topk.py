import torch

pred = torch.randn((4, 5))
print(pred)
values, indices = pred.topk(1, dim=1, largest=True, sorted=True)
print(indices)
# 用max得到的结果，设置keepdim为True，避免降维。因为topk函数返回的index不降维，shape和输入一致。
_, indices_max = pred.max(dim=1, keepdim=True)
print('end')
# print(indices_max == indices)
# # pred
# tensor([[-0.1480, -0.9819, -0.3364,  0.7912, -0.3263],
#         [-0.8013, -0.9083,  0.7973,  0.1458, -0.9156],
#         [-0.2334, -0.0142, -0.5493,  0.0673,  0.8185],
#         [-0.4075, -0.1097,  0.8193, -0.2352, -0.9273]])
# # indices, shape为 【4,1】,
# tensor([[3],   #【0,0】代表 第一个样本最可能属于第一类别
#         [2],   # 【1, 0】代表第二个样本最可能属于第二类别
#         [4],
#         [2]])
# # indices_max等于indices
# tensor([[True],
#         [True],
#         [True],
#         [True]])

'''
Date: 2021-07-21 11:37:38
LastEditors: Liuliang
LastEditTime: 2021-07-21 11:39:57
Description: 
'''
import torch
import torchvision
import numpy as np
dummy_input = torch.randn(1, 3, 224, 224,)
model = torchvision.models.resnet18(pretrained=True)

# 为输入输出起个名字
input_names = [ "input_node" ]
output_names = [ "output" ]

torch.onnx.export(model, dummy_input, "resnet18.onnx", verbose=True, input_names=input_names, output_names=output_names)


import onnxruntime as ort
dummy_input_numpy = dummy_input.numpy()
ort_session = ort.InferenceSession('resnet18.onnx')
outputs = ort_session.run(None, {'input_node': dummy_input_numpy.astype(np.float32)})

print(outputs[0])
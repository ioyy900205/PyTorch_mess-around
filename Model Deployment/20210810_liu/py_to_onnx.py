'''
Date: 2021-08-10 10:12:17
LastEditors: Liuliang
LastEditTime: 2021-08-10 15:19:34
Description: 
'''
import torch
import torchvision

# ================================================================== #
#                说明：torch to onnx                                             
# ================================================================== #	

model = torchvision.models.resnet152(pretrained=True)
batch_size = 1 
input_shape = (3, 224, 224)   
model.eval()
x = torch.randn(batch_size, *input_shape)   # 生成张量

export_onnx_file = "test.onnx"		# 目的ONNX文件名
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})



'''
Date: 2021-08-10 10:29:10
LastEditors: Liuliang
LastEditTime: 2021-08-10 15:24:01
Description: 
'''
 
import os, sys
sys.path.append(os.getcwd())
import onnxruntime
import onnx
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from PIL import Image

class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # image_numpy = image_numpy.transpose(2, 0, 1)
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        scores = self.onnx_session.run(self.output_name, input_feed=input_feed)[0]
        return scores



def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
# ================================================================== #
#                说明：cv2读图片                                             
# ================================================================== #	
r_model_path="test.onnx"
# img = cv2.imread("/home/liuliang/images/dog.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
# # img = cv2.resize(img, 224, 224)

# ================================================================== #
#                说明：Image PIL读图片                                             
# ================================================================== #	
data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        #  transforms.Normalize([1/255, 1/255, 1/255],  [0.229, 0.224, 0.225])]
         )
img_path = "/home/liuliang/images/cat.jpg"
img = Image.open(img_path)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

"""
# scipy.misc.imread 读取的图片数据是 RGB 格式
# cv2.imread 读取的图片数据是 BGR 格式
# PIL.Image.open 读取的图片数据是RGB格式
# 注意要与pth测试时图片读入格式一致
"""
to_tensor = transforms.ToTensor()
# img = to_tensor(img)
# img = img.unsqueeze_(0)  

resnet18_liu = ONNXModel(r_model_path)
out = resnet18_liu.forward(to_numpy(img))
out = out.argmax(axis=1)[0]
print(out)

# model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
# model.eval()
# output = model(img)


# val, cls = torch.max(output, 1)
# print("[pytorch]--->predicted class:", cls.item())
# print("[pytorch]--->predicted value:", val.item())

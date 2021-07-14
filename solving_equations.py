import torch  
from torch.autograd import Variable  
x=torch.Tensor([100.])  
#建立一个张量  tensor([1.], requires_grad=True)  
x=Variable(x,requires_grad=True)  
print('grad',x.grad,'data',x.data)  
learning_rate=0.01 
epochs=5000
  
for epoch in range(epochs):  
    y = x**2  
    y.backward()  
    print('grad',x.grad.data)  
    x.data=x.data-learning_rate*x.grad.data  
    #在PyTorch中梯度会积累假如不及时清零  
    x.grad.data.zero_()  
  
    print(x.data)  
    print(y)  
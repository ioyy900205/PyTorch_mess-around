import numpy as np
import torch

def myCrossEntropyLoss(x, label):
    loss = []

    for i, cls in enumerate(label):
        # 对应上面公式的 -x[class]
        x_class = -x[i][cls]
        # 对应上面公式的 log(sum)
        log_x_j = np.log( sum([ np.exp(j) for j in x[i] ]) )
        loss.append(x_class + log_x_j)

    return np.mean(loss)


x = np.array([
            [ 0.1545 , -0.5706, -0.0739 ],
            [ 0.2990, 0.1373, 0.0784],
            [ 0.1633, 0.0226, 0.1038 ]
        ])

# 分类标签
label = np.array([0, 1, 0])

print("my CrossEntropyLoss output: %.4f"% myCrossEntropyLoss(x, label))

loss = torch.nn.CrossEntropyLoss()
x_tensor = torch.from_numpy(x)
label_tensor = torch.from_numpy(label)
output = loss(x_tensor, label_tensor)
print("torch CrossEntropyLoss output: ", output)
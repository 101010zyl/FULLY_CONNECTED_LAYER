import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#target=
#input=
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=4)
        self.conv2 = torch.nn.Conv2d(10, 8, kernel_size=4)
        self.fc1 = nn.Linear(128, 256)#4x4x8
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 卷积-relu激活-maxpooling池化
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1) # -1自适应 0行数 ps：nn.Linear()结构，输入输出都是维度为一的值 view实现类似reshape的功能
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3
        return x

loss = torch.nn.MSELoss()
net = Net()
#optimizer = torch.optim.SGD(nnet.parameters(), lr=0.5, momentum=0.9);
print(net)
# for i in range(0,10):
#     optimizer.zero_grad()         #清空原有grad
#     out=net(input)                #前向传播
#     loss_value = loss(out,target) #损失计算
#     loss.backward()               #反向传播
#     optimizer.step()              #更新权值
#     if (i%100==0):
#         print(out)
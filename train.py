import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


#超参数的设置

epochs = 30
batch_size = 16
lr = 0.0001  # 防止忽略细节
dropout_possibility = 0.3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 30, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(30, 25, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(25, 20, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(20, 17, kernel_size=3)
        self.conv5 = torch.nn.Conv2d(17, 15, kernel_size=3)
        self.fc1 = nn.Linear(15*5*5, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout2d1 = nn.Dropout2d(p=dropout_possibility)
        self.dropout2d2 = nn.Dropout2d(p=dropout_possibility)
        self.dropout2d3 = nn.Dropout2d(p=dropout_possibility)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) # 卷积-relu激活-maxpooling池化
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout2d1(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = self.dropout2d2(x)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = self.dropout2d3(x)
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = x.view(batch_size, -1) # -1自适应 0行数 ps：nn.Linear()结构，输入输出都是维度为一的值 view实现类似reshape的功能
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


model = Net()

model.to(device)    #模型加载到设备上训练

img_valid_pth = "data/val"
imgs_train_pth = "data/train"

transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1)),  # 随机裁剪+resize
    transforms.RandomHorizontalFlip(),  # 随机翻转
    transforms.ToTensor(),  # rgb归一化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 使用论文的参数
])

dataset_to_train = datasets.ImageFolder(imgs_train_pth, transform)
dataset_to_valid = datasets.ImageFolder(img_valid_pth, transform)
print(dataset_to_train.class_to_idx)    # 打印 labels
# labels = torch.Tensor(labels).long()
train_loader = torch.utils.data.DataLoader(dataset_to_train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset_to_valid, batch_size=batch_size)




#优化器一类的
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)
criterion = nn.CrossEntropyLoss()

def train(epoch):
   # vis =True

   # i = 1
    loss_total = 0
    correct = 0
    #total = len(train_loader.dataset)
    model.train() # 进行训练
    #loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):   # 开始迭代
        data = Variable(data).to(device)
        target = Variable(target).to(device)
        targets = target.to(device, dtype=torch.float)
        #print(target.shape)
        #data = data.flatten()
        #data = np.reshape(data, 16)
        #print(data.shape)
        output = model(data)
        #print(output.shape)
        #output2 = torch.max(output,dim=1).values
        #print(output2.shape)
        #output2 = torch.cat((output2, output2), 0)
        #print(output2.shape)
        optimizer.zero_grad()
        _,predict_label = torch.max(output.data, 1)

        loss = criterion(output, targets.long())
        loss.backward()

        correct += torch.sum(predict_label == target)

        optimizer.step()

        loss_total += loss.data.item()
        # if (batch_idx + 1) % 5 == 0:
        #    print('Train : Epoch =  {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
        #               100. * (batch_idx + 1) / len(train_loader), loss.item()))
        # losses =  loss_total / len(train_loader)
        # i += 1
        # loop.set_description(f'Epoch [{epoch}/{epochs}]')
        # loop.set_postfix(loss=loss.item() / (batch_idx + 1), acc=correct / total)
        # print('\n', "train :  epoch =", epoch,
        # " learn rate =" , optimizer.param_groups[0]['lr'],
        # " loss =", losses /(batch_idx + 1) , " accuracy =", correct / total, '\n')
        if (batch_idx + 1) % 500 == 0:
            average_loss = loss_total / len(train_loader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
            #print('epoch:{},loss:{}'.format(epoch, average_loss))



def valid(model,valid_loader):
    loss_total = 0
    correct = 0
    total = len(valid_loader.dataset)
    model.eval()
    print(total, len(valid_loader))
    with torch.no_grad():

        for data, target in valid_loader:

            data, target = Variable(data).to(device), Variable(target).to(device)

            output = model(data)

            loss = criterion(output, target)

            _, pred = torch.max(output.data, 1)

            correct += torch.sum(pred == target)

            print_loss = loss.data.item()

            loss_total += print_loss

        correct = correct.data.item()

        accuracy = correct / total

        losses = loss_total / len(valid_loader)

        scheduler.step(print_loss / epoch + 1)

        print('validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            losses, correct, len(valid_loader.dataset), 100 * accuracy))

for epoch in range(1, epochs + 1):

    train(epoch)

    valid(model, valid_loader)

torch.save(model, 'model.pth')

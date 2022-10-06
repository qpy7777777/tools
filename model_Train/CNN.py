import torch
import torch.nn as nn
from torch.nn import functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)  # 1 指的是input channel，3指的是kernel数量
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,stride=1,padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        # self.avgPool = nn.AdaptiveAvgPool2d(1)
        # self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*5*5,2)  # [n+2p-f]/s
    def forward(self,x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        print("shape:",x.shape) #shape: torch.Size([3, 32, 5, 5])
        # x = self.avgPool(x)
        # x = self.flatten(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
net = CNN()
input = torch.rand(3,1,28,28)
print(input.shape)
print(net(input).shape)
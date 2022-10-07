# https://blog.csdn.net/qq_39567427/article/details/111936221
# https://blog.csdn.net/m0_57210162/article/details/120441275
# https://zhuanlan.zhihu.com/p/56638625
# https://blog.csdn.net/cxkyxx/article/details/108455805
import torch
from torch import nn
from torch.nn import functional as F
# from d2l import troch as d2l

class Residual(nn.Module):
    '''实现子modul：residualblock'''
    def __init__(self,input_channels,numn_channels,use_1x1Conv=False,strides=1):
        super(Residual, self).__init__()# 继承 nn.Module
        print(strides)
        self.conv1 = nn.Conv1d(input_channels,numn_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2 = nn.Conv1d(numn_channels,numn_channels,kernel_size=3,padding=1)

        if use_1x1Conv:
            self.conv3 = nn.Conv1d(input_channels,numn_channels,kernel_size=1,stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm1d(numn_channels)
        self.bn2 = nn.BatchNorm1d(numn_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y+=X
        return F.relu(Y)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = Residual(input_channels=64, numn_channels=64, strides=1)
        # self.conv3 = Residual(input_channels=64, numn_channels=128, use_1x1Conv=True,strides=2) #torch.Size([4, 128, 4])
        self.conv3 = Residual(input_channels=64, numn_channels=128, use_1x1Conv=True)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        out = self.conv2(out)
        out= self.conv3(out)
        out = self.pool(out)
        return out

class ResBlock1(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1):
        super(ResBlock1, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv1d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(out_channel)
        )
        self.shortcut=nn.Sequential()
        if in_channel != out_channel or stride>1:  #判断输入输出通道数是否匹配，经过卷积后原始图像是否减少，如不一致，需要将原始影像缩小至相同大小
            self.shortcut=nn.Sequential(
                nn.Conv1d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=stride,padding=1),
                nn.BatchNorm1d(out_channel)
            )

    def forward(self,x):
        output1=self.layer(x)
        output2=self.shortcut(x)
        output3=output1+output2
        output=F.relu(output3)
        return output

class ResCnn(nn.Module):
    def __init__(self):
        super(ResCnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = ResBlock1(in_channel=64, out_channel=64, stride=1)
        self.conv3 = ResBlock1(in_channel=64, out_channel=128, stride=1)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool(out)
        return out

# 输入和输出形状一致的情况
# blk = Residual(1,3)
X = torch.rand(4,1,32)
model = ResNet()
net = ResCnn()  #torch.Size([4, 128, 8])

# Y = blk(X)
print(model(X).shape)#torch.Size([4, 128, 4])
print(net(X).shape)#torch.Size([4, 128, 8])

# 增加输出通道数的同时，减半输出的高和宽。
# blk = Residual(1,6,use_1x1Conv=True,strides=2)
# print(blk(X).shape)
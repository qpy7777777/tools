import torch.nn as nn
import torch
from torch.nn import functional as F

class SELayer(nn.Module):
    def __init__(self,channel,reduction=16): #特征图通道的降低倍数16
        super(SELayer, self).__init__()
        # nn.AdaptiveAvgPool2d(1)
        # 维度变化：（2,512,8,8）-> (2,512,1,1) ==> (2,512)
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # 对 CxHxW 进行global average pooling
        # 得到 1x1xC大小的特征图，这个特征图可以理解为具有全局感受野
       # 使用一个全连接神经网络，对Sequeeze之后的结果做一个非线性变换。
        self.fc = nn.Sequential(
            nn.Linear(channel,channel // reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction,channel,bias=True),
            nn.Sigmoid()
        )
    def forward(self,x):
        # b,c,_,_ = x.size()   # (batch,channel,height,width) (2,512,8,8)
        b,c,_ = x.size()
        y = self.avg_pool(x).view(b,c) # 全局平均池化 (2,512,8,8) -> (2,512,1,1) -> (2,512)
        #y = self.fc(y).view(b, c, 1,1) ## (2,512) -> (2,512//reducation) -> (2,512) -> (2,512,1,1)
        y = self.fc(y).view(b,c,1)
        # 使用Excitation 得到的结果作为权重，乘到输入特征上。
        return x * y.expand_as(x) # (2,512,8,8)* (2,512,1,1) -> (2,512,8,8)

class ResBlock1(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,reduction=16):
        super(ResBlock1, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv1d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(out_channel)
        )
        self.se = SELayer(out_channel,reduction)
        self.shortcut=nn.Sequential()
        if in_channel != out_channel or stride>1:  #判断输入输出通道数是否匹配，经过卷积后原始图像是否减少，如不一致，需要将原始影像缩小至相同大小
            self.shortcut=nn.Sequential(
                nn.Conv1d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=stride,padding=1),
                nn.BatchNorm1d(out_channel)
            )

    def forward(self,x):
        output1=self.layer(x)
        # 注意力机制
        output1 = self.se(output1)

        output2=self.shortcut(x)
        output3=output1+output2
        output=F.relu(output3)
        return output

class SE_ResNet(nn.Module):
    def __init__(self):
        super(SE_ResNet, self).__init__()
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
X = torch.rand(4,1,32)
model = SE_ResNet()
print(model(X).shape)
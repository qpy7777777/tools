# nn.LSTM 参数
'''
input_size  输入数据的特征数量
hidden_size 输入数据的特征数
num_layer  构建的循环网络有几层的LSTM，默认是1
bias True 或者 FALSE 决定是否使用bisa
batch_first 默认false，将nn.lstm接受的数据输入x(序列长度,batch_size,输入特征数)->(batch_size,序列长度,输入特征数)
dropout
bidiretional 默认为FALSE，为TRUE时为双向lstm
'''
import torch
import torch.nn as nn

lstm = nn.LSTM(input_size=4, # 输入的特征数是4
               hidden_size=10,#输出的特征数
               batch_first=True #使用batch_first数据维度表达方式，即(batch_size,seq_len,feature_embding)

)
# 根据lstm网络的使用方式，每一层都有三个外界输入的数据，分别：
'''
X:LSTM 网络外输入的数据 (batch_szie,seq_len,dims)->(3,5,4)
h_0:上一层LSTM输出的结果 (num_layers * num_directions,batch_size,hidden_size)->(1,3,10)
c_0:上一层LSTM调整后的记忆 (num_layers * num_directions,batch_size,hidden_size)->(1,3,10)
'''
input = torch.randn(3,5,4)
# print(input)
# print("----------------")
h_0 = torch.randn(1,3,10)
c_0 = torch.randn(1,3,10)
# print(h_0,c_0)
# print("----------------")
# output = lstm(input,(h_0,c_0))
# print(output)
# print("----------------")
data,_ = lstm(input,(h_0,c_0))
out,(h_out,c_out) = lstm(input,(h_0,c_0))
print(out.shape) # (batch_size,序列长度，num_directions * hidden_size)
print(h_out.shape)
print(c_out.shape)
print(data.shape)
print(data[:,-1,:].shape) #torch.Size([3, 10])

# return out[:,-1,:] 或 [-1,:,:]




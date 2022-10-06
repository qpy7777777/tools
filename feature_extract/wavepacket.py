# 创建小波包
# 小波包能量 - python代码讲解
# https://blog.csdn.net/m0_47410750/article/details/125944236
import pywt
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
## 学会使用音频的小波变换系数进行训练\
# https://blog.csdn.net/weixin_48983346/article/details/125665507
# 基于python的信号小波分析 小波分解，小波重构，小波包分解，小波包重构
# https://blog.csdn.net/qq_34598178/article/details/106275346
# # print(pywt.families()) #打印出小波族
# # ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh',
# # 'morl', 'cgau', 'shan', 'fbsp', 'cmor']
# # for family in pywt.families(): #打印出每个小波族的每个小波函数
# #     print('%s family: '%(family) + ','.join(pywt.wavelist(family)))
# # db family: db1,db2,db3,db4,db5,db6,db7,db8,db9,db10,db11,db12,db13,
# # 创建小波包结构
# wp = pywt.WaveletPacket(data=audio,wavelet="db1",mode='symmetric')
# print(wp.data)
# #最大分解层数
# print(wp['ad'].maxlevel)
# #第一层：
# print(wp['a'].data)
# #第2 层
# print(wp['aa'].data)
# #第3 层时：
# print(wp['aaa'].data)
# #获取特定层数的所有节点
# print([node.path for node in wp.get_level(3, 'natural')]) #第3层有8个
# #依据频带频率进行划分
# print([node.path for node in wp.get_level(3, 'freq')])
#从小波包中重建数据
# x = [1, 2, 3, 4, 5, 6, 7, 8]
# wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')
# new_wp = pywt.WaveletPacket(data=None, wavelet='db1', mode='symmetric')
# new_wp['aa'] = wp['aa'].data
# new_wp['ad'] = wp['ad'].data
# new_wp['d'] = wp['d']
# print(new_wp.reconstruct(update=False))
# # 如果reconstruct方法中的update参数被设置为False，那么根节点的数据将不会被更新。
# print(new_wp.data)
# # 否则，根节点的data属性将被设置为重建后的数据。
# print(new_wp.reconstruct(update=True))
# print(new_wp.data)
# print([n.path for n in new_wp.get_leaf_nodes(False)])
# print([n.path for n in new_wp.get_leaf_nodes(True)])

def time_plot(audio,fs):
    librosa.display.waveplot(audio,fs)

def time_frequency(audio,fs):
    data = librosa.stft(audio)
    librosa.display.specshow(librosa.amplitude_to_db(abs(data)),x_axis="time",
                             y_axis='hz',sr=fs)
# Approximation and detail coefficients.
#正则化
def Normazition(data):
    return (data-min(data)) / (max(data)-min(data))

path = "../shipEar_classification/cut_data/4/81-1.wav"
audio,fs = librosa.load(path,sr=None)
cA,cD = pywt.dwt(audio,wavelet="db38", mode='smooth')
y1 = pywt.idwt(cA,None,wavelet="db38", mode='smooth')
y2 = pywt.idwt(None,cD,wavelet="db38", mode='smooth')
y3 = pywt.idwt(cA,cD,wavelet="db38", mode='smooth')
y_plot = [audio,y1,y2,y3]

title_plot = ["original","appro","detail","construct"]
figure = plt.figure(figsize=(12, 8))
cols,rows=2,2
for i in range(1, cols * rows + 1):
    figure.add_subplot(rows, cols, i)
    time_frequency(y_plot[i - 1],fs)
    plt.title(title_plot[i - 1])
    plt.tight_layout()
# plt.show()
figure2 = plt.figure(figsize=(12, 8))
for i in range(1, cols * rows + 1):
    figure2.add_subplot(rows, cols, i)
    time_plot(y_plot[i - 1],fs)
    plt.title(title_plot[i - 1])
    plt.tight_layout()
# plt.show()
# https://zhuanlan.zhihu.com/p/494060193
#根据频段频率（freq）进行排序
wp = pywt.WaveletPacket(data=audio, wavelet='db6',mode='symmetric',maxlevel=13) #选用db1小波，分解层数为3
#根据频段频率（freq）进行排序
print('第一层小波包节点:',[node.path for node in wp.get_level(1, 'freq')])  # 第一层小波包节点
print('第二层小波包节点:',[node.path for node in wp.get_level(2, 'freq')])  # 第二层小波包节点
print('第三层小波包节点:',[node.path for node in wp.get_level(3, 'freq')])  # 第三层小波包节点
# 提取分解系数：下面aaa是小波包变换第三层第一个的分解系数
aaa = wp['aaa'].data
print('aaa的长度:',aaa.shape[0])
print('audio的长度:',audio.shape[0])
print('理论上第3层每个分解系数的长度:',audio.shape[0]/pow(2,3))
# 绘制每一层小波分解时域图
# a = wp['a'].data #第1个节点
# d = wp['d'].data #第2个节点
# #第二层
# aa = wp['aa'].data
# ad = wp['ad'].data
# dd = wp['dd'].data
# da = wp['da'].data
# #第三层
# aaa = wp['aaa'].data
# aad = wp['aad'].data
# ada = wp['add'].data
# add = wp['ada'].data
# daa = wp['dda'].data
# dad = wp['ddd'].data
# dda = wp['dad'].data
# ddd = wp['daa'].data
# # 绘制小波图
# plt.figure(figsize=(15, 10))
# plt.subplot(4,1,1)
# plt.plot(audio)
# #第一层
# plt.subplot(4,2,3)
# plt.plot(a)
# plt.subplot(4,2,4)
# plt.plot(d)
# #第二层
# plt.subplot(4,4,9)
# plt.plot(aa)
# plt.subplot(4,4,10)
# plt.plot(ad)
# plt.subplot(4,4,11)
# plt.plot(dd)
# plt.subplot(4,4,12)
# plt.plot(da)
# #第三层
# plt.subplot(4,8,25)
# plt.plot(aaa)
# plt.subplot(4,8,26)
# plt.plot(aad)
# plt.subplot(4,8,27)
# plt.plot(add)
# plt.subplot(4,8,28)
# plt.plot(ada)
# plt.subplot(4,8,29)
# plt.plot(dda)
# plt.subplot(4,8,30)
# plt.plot(ddd)
# plt.subplot(4,8,31)
# plt.plot(dad)
# plt.subplot(4,8,32)
# plt.plot(daa)
# plt.show()
# 使用的wpd_plt(signal,n)将上面的代码优化和封装了，signal代表输入信号，n代表分解层数

def wpd_plt(signal, n):
    '''
    fun: 进行小波包分解，并绘制每层的小波包分解图
    param signal: 要分解的信号，array类型
    n: 要分解的层数
    return: 绘制小波包分解图
    '''
    # wpd分解
    wp = pywt.WaveletPacket(data=signal, wavelet='db6', mode='symmetric', maxlevel=3)
    # 计算每一个节点的系数，存在map中，key为'aa'等，value为列表
    map = {}
    map[1] = signal
    for row in range(1, n + 1):
        for i in [node.path for node in wp.get_level(row, 'freq')]:
            map[i] = wp[i].data
    # 作图
    plt.figure(figsize=(15, 10))
    plt.subplot(n + 1, 1, 1)  # 绘制第一个图
    plt.plot(map[1])
    for i in range(2, n + 2):
        level_num = pow(2, i - 1)  # 从第二行图开始，计算上一行图的2的幂次方
        # 获取每一层分解的node：比如第三层['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']
        re = [node.path for node in wp.get_level(i - 1, 'freq')]
        for j in range(1, level_num + 1):
            plt.subplot(n + 1, level_num, level_num * (i - 1) + j)
            plt.tight_layout()
            plt.plot(map[re[j - 1]])  # 列表从0开始

wpd_plt(signal=audio, n=3)

# 绘制小波包能量图
# wp = pywt.WaveletPacket(data=audio, wavelet='db6',mode='symmetric')
# print(f"wp.maxlevel{wp.maxlevel}")
wp = pywt.WaveletPacket(data=audio, wavelet='db6',mode='symmetric',maxlevel=3) #选用db1小波，分解层数为3
n = 3
re = []  #第n层所有节点的分解系数
for i in [node.path for node in wp.get_level(n, 'freq')]:
    re.append(wp[i].data)
#第n层能量特征
energy = []
for i in re:
    energy.append(pow(np.linalg.norm(i,ord=None),2))
for i in range(len(energy)):
    print('最后一层第{0}个小波的能量为：{1}'.format(i, energy[i]))
#绘制最后一层能量图
# 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
plt.figure(figsize=(10, 7), dpi=80)
# 再创建一个规格为 1 x 1 的子图
# plt.subplot(1, 1, 1)
# 柱子总数
N = 8
values = energy
# 包含每个柱子下标的序列
index = np.arange(N)
# 柱子的宽度
width = 0.45
# 绘制柱状图, 每根柱子的颜色为紫罗兰色
p2 = plt.bar(index, values, width, label="num", color="#87CEFA")
# 设置横轴标签
plt.xlabel('clusters')
# 设置纵轴标签
plt.ylabel('Wavel energy')
# 添加标题
plt.title('Cluster Distribution')
# 添加纵横轴的刻度
plt.xticks(index, ('7', '8', '9', '10', '11', '12', '13', '14'))
# plt.yticks(np.arange(0, 10000, 10))
# 添加图例
plt.legend(loc="upper right")
plt.show()
# 小波包能量集中在7频段，即在0-2198Hz范围内。
# # 关键程序理解
# map = {}
# map[1] = audio
# n = 3
# for row in range(1,n+1):
#     for i in [node.path for node in wp.get_level(row, 'freq')]:
#         print(i)
#         map[i] = wp[i].data
# # 由打印结果可知，i为每个小波包的名字
# print(map)
# # 可知map是个字典，key为小波包名，value为每个小波包的系数
# 重构音频
# x = [1,2,3,4,5,6,7,8]
# wp = pywt.WaveletPacket(data=x,wavelet="db1",mode="symmetric")
# print(wp.maxlevel)
# paths = [node.path for node in wp.get_level(3, 'freq')]
# print(paths)
# # a=[]
# # for i,path in enumerate(paths):
# #     data = wp[path].data
# #     a.append(data)
# # print(a)
# new_wp = pywt.WaveletPacket(data=None,wavelet="db1",mode="symmetric")
# nodeArr=np.array([node.path for node in wp.get_level(3, 'freq')])#获取第九层节点数组
# print(nodeArr)
# for j in range(0, 8):
#     cunrrentNode = nodeArr[j]
#     new_wp[cunrrentNode] = wp[cunrrentNode]
#     print(new_wp[cunrrentNode])
# newsign = new_wp.reconstruct(update=True)#4-8hz重构小波（θ 节律）
# print(newsign)


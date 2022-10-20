import pywt
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
## 学会使用音频的小波变换系数进行训练\

path = "../ShpsEar/cut_shipsEar/2.wav"
audio,fs = librosa.load(path,sr=None)
print(fs)
wp = pywt.wavedec(data=audio, wavelet='db6',mode='symmetric',level=5) #选用db1小波，分解层数为3

n = 6
re = []  #第n层所有节点的分解系数
for i in range(len(wp)):
    re.append(wp[i].data)
    print(len(wp[i].data))
#第n层能量特征
energy = []
for i in re:
    energy.append(pow(np.linalg.norm(i,ord=None),2))
for i in range(len(energy)):
    print('最后一层第{0}个小波的能量为：{1}'.format(i, energy[i]))
#绘制最后一层能量图
# 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
plt.figure(figsize=(10, 7), dpi=80)
N = 6
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
plt.xticks(index, ('1', '2', '3', '4',"5","6"))
# plt.xticks(index, ('7', '8', '9', '10', '11', '12', '13', '14'))
# plt.yticks(np.arange(0, 10000, 10))
# 添加图例
plt.legend(loc="upper right")
plt.show()



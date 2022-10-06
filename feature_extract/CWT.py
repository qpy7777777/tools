import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.fftpack import fft
import pywt
# 学会使用音频的小波变换系数进行训练\
# https://blog.csdn.net/weixin_48983346/article/details/125665507

# print(len(data),fs)
# data = data[:44100]
# print(fs,len(data))
#
# t = np.arange(0,len(data)/fs,1.0 / fs)
# librosa.display.waveplot(data, sr=fs)
# plt.show()
#
# axix_x = np.linspace(0,1,len(data))
# Y1 = abs(fft(data))
# plt.plot(range(fs), Y1)
# plt.title('signal_1 in time domain')
# plt.xlabel('Time/second')
# plt.show()
#
# melspec = librosa.feature.melspectrogram(data, fs, n_fft=1024, hop_length=512, n_mels=128)
# librosa.display.specshow(librosa.power_to_db(melspec),x_axis='time',y_axis='mel',sr=fs)
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
# plt.tight_layout()
# plt.show()
#
# #连续小波变化返回的是小波系数，尺度频率；离散小波变换返回的是近似系数、细节系数。
# 高频系数 D （细节部分）和低频系数 A （近似部分）
# def cwt(x, fs, totalscal, wavelet='cgau8'):
#     if wavelet not in pywt.wavelist():
#         print('小波函数名错误')
#     else:
#         wfc = pywt.central_frequency(wavelet=wavelet)
#         a = 2 * wfc * totalscal / np.arange(totalscal, 1, -1)
#         period = 1.0 / fs
#         [cwtmar, fre] = pywt.cwt(x, a, wavelet, period)
#         cwtmar = abs(cwtmar)
#         return cwtmar, fre
#
def dwt(x, wavelet='db3'):
    cA, cD = pywt.dwt(x, wavelet, mode='symmetric')
    ya = pywt.idwt(cA, None, wavelet, mode='symmetric')
    yd = pywt.idwt(None, cD, wavelet, mode='symmetric')
    return ya, yd, cA, cD
#
# totalscal = 256
#
# cwtmar, fre = cwt(data,fs,totalscal,'cgau8')
# plt.contourf(t, fre, cwtmar)
# plt.ylabel("Hz")
# plt.xlabel("time")
# plt.show()
#
path = "../original/0/7061-6-0-0.wav"
data, fs = librosa.load(path,sr=None)
t = np.arange(0,len(data)/fs,1/fs)
print(len(t))
ya,yd,ca,cd = dwt(data,'db3')
print(len(ya))
# 离散小波
plt.figure(figsize=(12,9))
plt.subplot(311)
plt.plot(t, data)
plt.title('original signal')
plt.subplot(312)
plt.plot(t, ya)
plt.title('approximated component')
plt.subplot(313)
plt.plot(t, yd)
plt.title('detailed component')
plt.tight_layout()
plt.show()

def wavelet_trans(self, signal):
    # 读取音频数据
    Signal = signal
    # [cA_n, cD_n, cD_n-1, ..., cD2, cD1]
    coeffs = pywt.wavedec(Signal, 'db1', level=3)#多层离散小波变换
    cA3, cD3, cD2, cD1 = coeffs
    xishu = np.append(cA3, cD3)
    xishu = np.append(xishu, cD2)
    signal = np.append(xishu, cD1)
    return signal
# 小波分解
# data=[1,2,3,4,5,6,7,8]
# wp = pywt.WaveletPacket(data = data,wavelet="db1",mode="symmetric",maxlevel=3)
# path = [node.path for node in wp.get_level(3,'freq')]
# list1 = []
# # 重构
# for i,path in enumerate(path):
#     list1.append(wp[path].data)
# print(list1)
# print(path)
# print(wp.data)
# print(wp.reconstruct())
# # 小波包树的节点由路径标识。标识根节点的路径是’ ',根节点的分解层数为0。
# print(repr(wp.path))
# print(wp.level)
# #关于最大分解层数，如果构造函数中没有指定参数，则自动计算。
# print(wp['ad'].maxlevel)
# #下面开始获取小波包树的子节点：
# print(wp['a'].data)
# print(wp['aa'].data)
# print(wp['aaa'].data)
# print(wp['aaa'].path)
print(pywt.families()) #打印出小波族
# for family in pywt.families(): #打印出每个小波族的每个小波函数
#     print('%s family: '%(family) + ','.join(pywt.wavelist(family)))
# db3 = pywt.Wavelet('db3') #创建一个小波对象
# print(db3)
# Family name:    Daubechies 小波函数
# Short name:     db 缩写名
x= [3, 7, 1, 1, -2, 5, 4, 6]

cA, cD = pywt.dwt(x, 'db2') #得到近似值和细节系数
# print(cA)
# print(cD)
print(pywt.idwt(cA, cD, 'db2'))

#传入小波对象，设置模式
w = pywt.Wavelet('sym3') # 创建小波对象
path = "../original/0/7061-6-0-0.wav"
data, fs = librosa.load(path,sr=None)
cA, cD = pywt.dwt(x, wavelet=w, mode='constant')
print(pywt.Modes.modes)
# print(pywt.idwt([1,2,0,1], None, 'db3', 'symmetric'))
# print(pywt.idwt([1,2,0,1], [0,0,0,0], 'db3', 'symmetric'))
#小波包 wavelet packets
X = data[2000:2010]
print(X)
wp = pywt.WaveletPacket(data=X, wavelet='db3', mode='symmetric', maxlevel=3)
#根据频段频率（freq）进行排序
print('第一层小波包节点:',[node.path for node in wp.get_level(1, 'freq')])  # 第一层小波包节点
print('第二层小波包节点:',[node.path for node in wp.get_level(2, 'freq')])  # 第二层小波包节点
print('第三层小波包节点:',[node.path for node in wp.get_level(3, 'freq')])  # 第三层小波包节点

print(wp)
print(wp.data)
print(repr(wp.path))
print(wp.level) # 0 #分解级别为0
# print(wp['ad'].maxlevel) # 3
# print(wp['a'].data)
# #第2 层
# print(wp['aa'].data)
# #第3 层时：
# print(wp['aaa'].data)
#从小波包中 重建数据
X = [1, 2, 3, 4, 5, 6, 7, 8]
wp = pywt.WaveletPacket(data=X, wavelet='db1', mode='symmetric', maxlevel=3)
print(wp['ad'].data) # [-2,-2]
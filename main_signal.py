import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pywt
import numpy as np
from matplotlib.font_manager import FontProperties
import os
font = FontProperties("SimHei")  # 定义一个字体对象
def time_plot(audio,fs):
    librosa.display.waveplot(audio,fs)
def spectrogram(audio,fs):
    data = librosa.stft(audio)
    librosa.display.specshow(librosa.power_to_db(abs(data)),x_axis="time",y_axis="hz",sr=fs)
    plt.colorbar(format='%+2.0f dB')
    # plt.set_cmap("winter")
def magnitude_scalogram(audio,fs):
    t = len(audio) / fs
    t = np.arange(0, t, 1.0 / fs)
    wavename = "cgau8"
    totalscale = 32
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscale
    scales = cparam / np.arange(totalscale,1,-1)
    [cwtmatr,frequencies] = pywt.cwt(audio,scales,wavename,1.0/fs)
    plt.contourf(t, frequencies, abs(cwtmatr),cmap="winter") # t = np.arange(0, 1.0, 1.0 / sampling_rate)
def wpd_plt(signal, n):
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
def dwt(X):
    cA, cD = pywt.dwt(X, wavelet="db6", mode='smooth')
    y1 = pywt.idwt(cA, None, wavelet="db6", mode='smooth')
    y2 = pywt.idwt(None, cD, wavelet="db6", mode='smooth')
    y3 = pywt.idwt(cA, cD, wavelet="db6", mode='smooth')
    y_plot = [X, y1, y2, y3]
    title_plot = ["original_signale", "appro", "detail", "construct"]
    figure = plt.figure(figsize=(12, 8))
    cols, rows = 2, 2
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        spectrogram(y_plot[i - 1], fs)
        plt.title(title_plot[i - 1])
        plt.tight_layout()
    plt.suptitle("离散小波", fontproperties=font, fontsize=20)  # 添加整个图表的标题
    plt.tight_layout()
    plt.show()
    figure2 = plt.figure(figsize=(12, 8))
    for i in range(1, cols * rows + 1):
        figure2.add_subplot(rows, cols, i)
        time_plot(y_plot[i - 1], fs)
        plt.title(title_plot[i - 1])
        plt.tight_layout()
    plt.show()
if __name__=="__main__":
    path = "ShpsEar/cut_shipsEar/car.wav"
    font = FontProperties("SimHei")  # 定义一个字体对象
    X,fs = librosa.load(path,sr=None)
    print(f'音频名称为{path},音频采样率为{fs}')
    # 时域波形图
    X = X[:44100]
    time_plot(X,fs)
    plt.title(os.path.basename(path))
    plt.suptitle("时域波形图", fontproperties=font, fontsize=20)  # 添加整个图表的标题
    plt.tight_layout()
    plt.show()
    # 谱图
    spectrogram(X, fs)
    plt.title(os.path.basename(path))
    plt.suptitle("时频图", fontproperties=font, fontsize=20)  # 添加整个图表的标题
    plt.tight_layout()
    plt.show()
    magnitude_scalogram(X, fs)
    plt.title(os.path.basename(path))
    plt.suptitle("小波图", fontproperties=font, fontsize=20)  # 添加整个图表的标题
    plt.tight_layout()
    plt.show()
    # 离散小波
    dwt(X)
    # 绘制小波包能量图
    wpd_plt(X, n=3)
        # 绘制小波包能量图
        # wp = pywt.WaveletPacket(data=audio, wavelet='db6',mode='symmetric')
        # print(f"wp.maxlevel{wp.maxlevel}")
    wp = pywt.WaveletPacket(data=X, wavelet='db1', mode='symmetric', maxlevel=3)  # 选用db1小波，分解层数为3
    aaa = wp['aaa'].data
    print('aaa的长度:', aaa.shape[0])
    print('audio的长度:', X.shape[0])
    print('理论上第3层每个分解系数的长度:', X.shape[0] / pow(2, 3))
    n = 3
    re = []  # 第n层所有节点的分解系数
    for i in [node.path for node in wp.get_level(n, 'freq')]:
        re.append(wp[i].data)
    # 第n层能量特征
    energy = []
    for i in re:
        energy.append(pow(np.linalg.norm(i, ord=None), 2))
    for i in range(len(energy)):
        print('最后一层第{0}个小波的能量为：{1}'.format(i, energy[i]))
    # 绘制最后一层能量图
    # 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
    plt.figure(figsize=(10, 7), dpi=80)
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








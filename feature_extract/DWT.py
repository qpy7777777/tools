import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,ifft
import pywt
from skimage.restoration import denoise_wavelet
# import numpy.fft as fft
plt.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False#用来正常显示符号
def time_plot(audio,fs):
    librosa.display.waveplot(audio,fs)
    plt.show()
'''
x:  采样的数据
dt=1/fs : 每样本的时间或空间增量(如果是时间上的增量，则又称：采样间隔或采样步长，单位常用:s);
t=(0:n-1)/fs : 数据的时间或空间范围;
y=fft(x) : 数据的离散傅里叶变换(DFT);
abs(y) : DFT的振幅;
(abs(y).^2)/n : DFT的幂;
fs/n : 频率增量;
f=(0:n-1) * (fs/n) : 频率范围;
fs/2 : Nyquist频率(频率范围的中点);
采样定理：采样频率要大于信号频率的两倍
假设采样频率为Fs，采样点数为N。那么FFT运算的结果就是N个复数（或N个点）
，每一个复数就对应着一个频率值以及该频率信号的幅值和相位。
第一个点对应的频率为0Hz（即直流分量），最后一个点N的下一个点对应采样频率Fs。其中任意一个采样点n所代表的信号频率：
Fn = (n-1)*Fs/N
'''
def frequency(audio,fs):
    # http: // www.javashuo.com / article / p - ktkuqtwf - ny.html
    # t = np.arange(0, 1.0, 1.0 / fs)# 1s时间取样  t = np.linspace(0,1,fs) # 生成 1s 的时间序列
    # L = round(len(audio) / 2)
    # 归一化操作
    # 通常有N个点做FFT就有 N 个复数点与之对应，此时幅值是对复数取绝对值运算。
    # 除第一个点之外，实际的幅值是信号的实际长度L，再乘以2。所以一般对信号先做FFT
    # 再取绝对值然后除以L乘以2
    L = len(audio)
    fuzhi = abs(fft(audio[0:L])) / L * 2
    # f = (0:n-1) * (fs / n): 频率范围;
    pinlv = (np.arange(0, L) / L ) * fs
    plt.plot(pinlv[1:int(L/2)+1],fuzhi[1:int(L/2)+1])
    plt.show()
def time_frequency(audio,fs):
    data = librosa.stft(audio)
    librosa.display.specshow(librosa.amplitude_to_db(abs(data)),x_axis="time",
                             y_axis='hz',sr=fs)
    plt.show()
def mel_specgram(audio,fs):
    mel_specgram = librosa.feature.melspectrogram(audio,sr=fs,n_fft=1024, hop_length=512,
                                              n_mels=128)
    librosa.display.specshow(librosa.power_to_db(mel_specgram),x_axis='time',y_axis='hz',sr=fs)
    plt.show()
def cwt(data,fs,totalcaler,wavelet):
    time = len(data) / fs
    t = np.arange(0,time,1/fs)
    # 中心频率
    fc = pywt.central_frequency(wavelet)
    # 计算对应频率的小波尺度
    cparam = 2 * fc * totalcaler
    scalers = cparam / np.arange(totalcaler,0,-1)
    [cwtmatr, frequencies] = pywt.cwt(data, scalers, wavelet, 1.0 / fs)
    cwtmatr = abs(cwtmatr)
    plt.contourf(t, frequencies, cwtmatr)
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.show()
# 单级分解
def dwt(data,x):
    wavename = 'db5'
    cA, cD = pywt.dwt(data,wavename)
    # 重构信号
    y = pywt.idwt(cA,cD,wavename)
    # 低频成分
    ya = pywt.idwt(cA, None, wavename,'smooth') # approximated component
    # 高频成分
    yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed componen
    plt.figure(figsize=(12,9))
    plt.subplot(311)
    plt.plot(x, y)
    plt.title('original signal')
    plt.subplot(312)
    plt.plot(x, ya)
    plt.title('approximated component')
    plt.subplot(313)
    plt.plot(x, yd)
    plt.title('detailed component')
    plt.tight_layout()
    plt.show()
# 多级分解
def dwtc(data,x):
    coeffs = pywt.wavedec(data,'db1',level=2,mode="periodic")
    cA2,cD2,cD1 = coeffs
    y = pywt.waverec(coeffs,'db1',mode="periodic")
    # 小波重构
    ya4 = pywt.waverec(np.multiply(coeffs,[1, 0, 0]).tolist(),'db1')
    plt.figure(figsize=(12, 9))
    plt.plot(x, y,'r')
    plt.plot(x, ya4,'b')
    plt.show()

def devoise(data,t):
    x_denoise = denoise_wavelet(data,method="VisuShrink", mode="soft",wavelet_levels=3,wavelet="sym8",rescale_sigma="True")
    plt.figure(figsize=(20,10),dpi=100)
    plt.plot(t, data, 'g')
    plt.plot(t, x_denoise, 'r')
    plt.show()

def Nomalization(data):
    data = (data - min(data)) / (max(data)-min(data))
    return data

def denoise_wavelet(data):
    w = pywt.Wavelet("db8")
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.04  # Threshold for filtering
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解
    print(coeffs[0].shape)
    print(len(coeffs))
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波

    training_set_scaled = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
    plt.plot(training_set_scaled, "b--")
    plt.plot(data,'r--')
    plt.show()

if __name__ == "__main__":
    # 音频路径
    path = "../ShipsEar/0.wav"
    audio, fs = librosa.load(path, sr=None)
    print(f"信号采样率{fs},信号持续时长{len(audio)/fs}")
    time = len(audio) / fs
    x = np.arange(0, time, 1 / fs)
    # time_plot(audio,fs)
    # frequency(audio,fs)
    # time_frequency(audio,fs)
    # mel_specgram(audio,fs)
    # wavename = 'cgau8'
    # totalscal = 256
    # # cwt(audio,fs,totalscal,wavename)
    # # dwt(audio,x)
    # dwtc(audio,x)
    denoise_wavelet(audio)
    # audio = Nomalization(audio)
    # devoise(audio,x)





import matplotlib.pyplot as plt
import numpy as np
import pylab as pl#导入一个绘图模块，matplotlib下的模块
import librosa.display
# 求幅值 乘上后面的2/N得到正确幅值
audio,fs = librosa.load("../ShpsEar/cut_shipsEar/0.wav",sr=None)
print(len(audio),fs,len(audio)/fs)
#np.arange(起点，终点，间隔)产生1s长的取样时间
t = np.arange(0, len(audio)/fs, 1.0/fs)
xf = np.fft.rfft(audio)/len(audio)
# 利用np.fft.rfft()进行FFT计算，rfft()是为了更方便对实数信号进行变换，
# 由公式可知/fft_size为了正确显示波形能量

# rfft函数的返回值是N/2+1个复数，分别表示从0(Hz)到sampling_rate/2(Hz)的分。
#于是可以通过下面的np.linspace计算出返回值中每个下标对应的真正的频率：
freqs = np.linspace(0, fs//2, len(audio)//2 + 1)
# 幅值
amplitude = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
# amplitude = 20*np.log10(np.abs(xf))
#最后我们计算每个频率分量的幅值，并通过 20*np.log10()将其转换为以db单位的值。
# 为了防止0幅值的成分造成log10无法计算，我们调用np.clip对xf的幅值进行上下限处理

#绘图显示结果
pl.figure(figsize=(8,4))
pl.subplot(211)
pl.plot(t, audio)
pl.xlabel(u"Time(S)")
pl.title(u"WaveForm")
pl.subplot(212)
# 对频率和幅值作图，xlabel是频率Hz,ylabel是dB
pl.plot(freqs,amplitude)
pl.xlabel(u"Freq(Hz)")
pl.subplots_adjust(hspace=0.4)
pl.show()

# 语谱图的横坐标是时间，纵坐标是频率，坐标点值为语音数据能量。由于是采用二维平面表达三维信息，
# 所以能量值的大小是通过颜色来表示的，颜色深，表示该点的语音能量越强。其可以理解为利用二维坐标表示三维信息。
# D = librosa.stft(audio)
# librosa.display.specshow(librosa.amplitude_to_db(np.abs(D)),x_axis="time",y_axis="hz",sr=fs)
# plt.colorbar()
# pl.show()

# sampling_rate = 8000  # 采样频率8000Hz
# fft_size = 512  # 采样点512，就是说以8000Hz的速度采512个点，我们获得的数据只有这512个点的对应时刻和此时的信号值。
# t = np.linspace(0, 1, sampling_rate)  # 截取一段时间，截取是任意的，这里取了0~1秒的一段时间。
#
# x = np.sin(2 * np.pi * 156.25 * t) + 2 * np.sin(2 * np.pi * 234.375 * t)  # 输入信号序列，人工生成了一段信号序列，范围在0~1秒
# xs = x[:fft_size]  # 由上所述，我们只采样了512个点，所以我们只获得了前512个点的数据
# xf = np.fft.rfft(
#     xs) / fft_size  # 调用np.fft的函数rfft(用于实值信号fft)，产生长度为fft_size/2+1的一个复数向量，分别表示从0Hz~4000Hz的部分，这里之所以是4000Hz是因为Nyquist定理，采样频率8000Hz，则能恢复带宽为4000Hz的信号。最后/fft_size是为了正确显示波形能量
#
# freqs = np.linspace(0, sampling_rate // 2, fft_size // 2 + 1)  # 由上可知，我们得到了数据，现在产生0~4000Hz的频率向量，方便作图
# xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e1000))  # 防止幅值为0，先利用clip剪裁幅度，再化成分贝
#
#
# pl.figure(figsize=(8, 4))  # 生成画布
# pl.subplot(211)  # 生成子图，211的意思是将画布分成两行一列，自己居上面。
# pl.plot(t[:fft_size], xs)  # 对真实波形绘图
# pl.xlabel(u"time(s)")
# pl.title(u"The Wave and Spectrum of 156.25Hz and 234.375Hz")
# pl.subplot(212)  # 同理
# pl.plot(freqs, xfp)  # 对频率和幅值作图，xlabel是频率Hz,ylabel是dB
# pl.xlabel(u"Hz")
# pl.subplots_adjust(hspace=0.4)  # 调节绘图参数
# pl.show()
# """
# 功率谱 power spectrum
# 直接平方
# """
ps = np.abs(xf)**2
ax=plt.subplot(513)
ax.set_title('direct method')
plt.plot(freqs,20*np.log10(ps))

"""
相关功谱率 power spectrum using correlate
间接法
"""
cor_x = np.correlate(audio, audio, 'same')
cor_X = np.fft.rfft(cor_x)/len(audio)
ps_cor = np.abs(cor_X)
ps_cor = ps_cor / np.max(ps_cor)
ax=plt.subplot(514)
ax.set_title('indirect method')
plt.plot(freqs,20*np.log10(ps_cor))
plt.tight_layout()
plt.show()








#  功率谱
# 傅里叶直接平方和/序列长
# ps = pow(xf, 2) / len(audio)
# plt.plot(freqs,20*np.log10(ps))
# # "相关功谱率 power spectrum using correlate间接法"""
# cor_x = np.correlate(audio, audio, 'same')
# cor_X = np.fft.fft(cor_x)
# ps_cor = np.abs(cor_X)
# ps_cor = ps_cor / np.max(ps_cor)
# plt.title('indirect method')
# plt.plot(20*np.log10(ps_cor))
# plt.tight_layout()


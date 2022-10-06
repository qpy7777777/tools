import os.path
import soundfile
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
# min-max归一化
def Normalization(data):
    data = (data-min(data))/(max(data)-min(data))
    return data
# 提取音频特征
#可以看到重建的语音，与原本的语音的波形图虽然大致的波形相同，但是在许多细节上不一样。
def audio_mfcc(audio,fs,n_mfcc):
    MFCCs = librosa.feature.mfcc(audio,sr=fs,n_mfcc=n_mfcc)
    return MFCCs
# 音频重构
def mfcc_audio(mfccs):
    # 将MFCC转音频
    y = librosa.feature.inverse.mfcc_to_audio(mfccs)
    return y
# 提取音频mel特征
def mel_feature(data,fs,n_mel):
    spec = librosa.feature.melspectrogram(y=data,sr=fs,n_fft=2048,hop_length=512,win_length=None,
                                          window='hann',
                                          center=True,
                                          pad_mode='reflect',
                                          power=2.0,
                                          n_mels=n_mel)
    return spec
# 音频mel特征重构
def mel_audio(data,fs):
    res = librosa.feature.inverse.mel_to_audio(data,sr=fs,n_fft=2048, hop_length=512,
                                               win_length=None,
                                               window='hann',
                                               center=True,
                                               pad_mode='reflect',
                                               power=2.0,
                                               n_iter=32)
    return res
# 时域波形图
def plot_time(data,fs,title):
    # title = ["origin audio", "mfcc_20 reconsitution","mfcc_40 reconsitution","mfcc_100 reconsitution"]
    rows, cols = 2, 2
    figure = plt.figure(figsize=(12, 8))
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        librosa.display.waveplot(data[i - 1], sr=fs)
        plt.title(title[i - 1])
        plt.tight_layout()
        plt.savefig("mel_com.pdf")
    plt.show()
# 欧氏距离
def calEuclidean(x, y):
    dist = np.sqrt(np.sum(np.square(x-y)))   # 注意：np.array 类型的数据可以直接进行向量、矩阵加减运算。np.square 是对每个元素求平均~~~~
    return dist

def calPSNR(x,y):
    psnrNoise = peak_signal_noise_ratio(x, y)
    return psnrNoise

def plot_spec(data,fs,title):
    rows, cols = 2, 2
    figure = plt.figure(figsize=(12, 8))
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        amp = librosa.stft(data[i-1])
        librosa.display.specshow(librosa.power_to_db(abs(amp)), sr=fs,x_axis='time', y_axis='mel')
        plt.title(title[i - 1])
        plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    select = "False"
    select_mel = "False"
    if select == "True":
        audio = "./data_aug/15-1_0.wav"
        audio, fs = librosa.load(audio, sr=None)
        for n_mfcc in ["20",'40','100']:
            mfcc = audio_mfcc(audio,fs,int(n_mfcc))
            audio = mfcc_audio(mfcc)
            path = os.path.join("./construct",n_mfcc+'.wav')
            soundfile.write(path, audio, fs)  # 注意：只能写成WAV格式的
    elif select_mel == "True":
        audio = "./data_aug/15-1_0.wav"
        audio, fs = librosa.load(audio, sr=None)
        for n_mel in ["64", '128', '256']:
            mfcc = audio_mfcc(audio, fs, int(n_mel))
            audio = mfcc_audio(mfcc)
            path = os.path.join("./mel_construct", n_mel + '.wav')
            soundfile.write(path, audio, fs)  # 注意：只能写成WAV格式的
    else:
        path = "./mel_construct"
        audio = []
        title = []
        file = os.listdir(path)
        for i in range(len(file)):
            data,fs = librosa.load(os.path.join(path,file[i]),sr=None)
            data = Normalization(data)
            audio.append(data)
            filename = file[i]
            title.append(filename)
            print(title)
        plot_time(audio,fs,title)
        plot_spec(audio,fs,title)
        dist1 = calEuclidean(audio[1][:157696],audio[0])
        dist2 = calEuclidean(audio[1][:157696], audio[2])
        dist3 = calEuclidean(audio[1][:157696], audio[3])
        print(f"{title[1]}和{title[0]}的距离为{dist1}")
        print(f"{title[1]}和{title[2]}的距离为{dist2}")
        print(f"{title[1]}和{title[3]}的距离为{dist3}")





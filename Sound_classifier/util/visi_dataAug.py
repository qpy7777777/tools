import librosa
import librosa.display
import matplotlib.pyplot as plt
import time
import os
import numpy as np

# min-max归一化
def Normalization(data):
    data = (data-min(data))/(max(data)-min(data))
    return data
# 时域波形图
def plot_time(data,fs,title):
    # title = ["origin audio", "mfcc_20 reconsitution","mfcc_40 reconsitution","mfcc_100 reconsitution"]
    rows, cols = 2, 3
    figure = plt.figure(figsize=(12, 8))
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        librosa.display.waveplot(data[i - 1], sr=fs)
        plt.title(title[i - 1])
        plt.tight_layout()
        plt.savefig("data_augment.pdf")
    plt.show()

if __name__ == "__main__":
        path = "./data_aug"
        audio = []
        title = ["orginal audio","audio_SNR","audio_noise","audio_pitchShift"
            ,"audio_timeStreching","audio_timeShift"]
        file = os.listdir(path)
        for i in range(len(file)):
            data,fs = librosa.load(os.path.join(path,file[i]),sr=None)
            # data = Normalization(data)
            audio.append(data)
            # filename = file[i]
            # title.append(filename)
            print(title)
        plot_time(audio,fs,title)


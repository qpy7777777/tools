import librosa.display
#.waw文件转声谱图
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
genres = '0 1 2 3 4'.split()

labels_map = {
    0: "ship signal",
    1: "background noise",
    2: "humpback whale",
    3: "pilot whale",
    4: "bowhead whale",
}
cols,rows = 3,2
# figure = plt.figure(figsize=(8, 6))
#
# for g in genres:
#     pathlib.Path(f'image/{g}').mkdir(parents=True, exist_ok=True)
#     for filename in os.listdir(f'original/{g}'):
#         audioname = f'original/{g}/{filename}'
#         y, sr = librosa.load(audioname,sr=None)
#         X = librosa.stft(y)
#         Xdb = librosa.amplitude_to_db(abs(X))
#         figure.add_subplot(cols, rows, int(g) + 1)
#         librosa.display.specshow(Xdb, sr=sr,x_axis="time",y_axis="hz")
#         # fig = plt.gcf()
#         # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
#         # plt.margins(0,0)
#         # plt.colorbar(format='%+2.0f dB')
#         # plt.title(f'{labels_map[int(g)]} Spectrogram')
#         # plt.show()
#         plt.title(f'{labels_map[int(g)]} Spectrogram')
#         plt.tight_layout()


for g in genres:
    pathlib.Path(f'image/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'original/{g}'):
        audioname = f'original/{g}/{filename}'
        y, sr = librosa.load(audioname,sr=None)
        X = librosa.stft(y)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb, sr=sr,x_axis="time",y_axis="hz")
        fig = plt.gcf()
        # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        # plt.colorbar(format='%+2.0f dB')
        plt.title(f'{labels_map[int(g)]} Spectrogram')
        plt.show()
    fig.savefig(f'image/{g}/{filename[:-3].replace(".", "")}.pdf')

print("finish")

# # path = "data/0/data_CH1_1.wav"
# # audio,fs = librosa.load(path,sr=10000)
# # D = librosa.stft(audio)
# # Xdb = librosa.amplitude_to_db(abs(D))
# # print(Xdb.shape)
# # librosa.display.specshow(Xdb,sr=fs)
# # fig = plt.gcf()
# # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
# # plt.margins(0,0)
# # fig.savefig("fig.jpg")
# # plt.show()


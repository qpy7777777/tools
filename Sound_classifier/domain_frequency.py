import os
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from scipy.signal import periodogram
import seaborn as sns

# 查看据集类别主频率的分布图

freqs = {k: [] for k in range(5)}
path = os.listdir(r"underwater/train")
print(path)
path= sorted(path)
print(path)
for folder in path:
    print(folder)
    for file in os.listdir(r"underwater/train" + '/' + folder):
        class_id = os.path.basename(folder)
        data, sample_rate = librosa.load(r"underwater/train" + '/' + folder + '/' + file, sr=None)
        freq, PSD = periodogram(data, fs=sample_rate)
        max_id = np.flip(np.argsort(PSD))[0]
        freqs[int(class_id)].append(freq[max_id])

    print('------Done for a folder..!----')

# for class 0
sns.displot(freqs[0], kde=True, color="#098154")
plt.title("Ship_sign's main frequency density map")
plt.savefig("ship.pdf")

# for class 1
sns.displot(freqs[1], kde=True, color="dodgerblue")
plt.title("Background_noise's main frequency density map")
plt.savefig("Background_noise.pdf")

# for class 2
sns.displot(freqs[2], kde=True, label='Humpback whales', color="orange")
plt.title("Humpback_whale's main frequency density map")
plt.savefig("Humpback_whales.pdf")

# for class 3
sns.displot(freqs[3], kde=True, label='Pilot whales', color="deeppink")
plt.title("Pilot_whale's main frequency density map")
plt.savefig("Pilot_whales.pdf")

# for class 4
sns.displot(freqs[4], kde=True, label='Bowhead whales', color="saddlebrown")
plt.title("Bowhead_whale's main frequency density map")
plt.savefig("Bowhead_whales.pdf")

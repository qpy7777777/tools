# # 画图
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa
import os
# def add_subplot_axes(ax, position):
#     box = ax.get_position()
#     position_display = ax.transAxes.transform(position[0:2])
#     position_fig = plt.gcf().transFigure.inverted().transform(position_display)
#     x = position_fig[0]
#     y = position_fig[1]
#
#     return plt.gcf().add_axes([x, y, box.width * position[2], box.height * position[3]], fc='w')
# def plot_clip_overview(clip, ax):
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     ax_waveform = add_subplot_axes(ax, [0.0, 0.7, 1.0, 0.3])
#     ax_spectrogram = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.7])
#     audio, sample_rate = librosa.load(clip, sr=None)
#     ax_waveform.plot(np.arange(0, len(audio)) / sample_rate, audio)
#     ax_waveform.get_xaxis().set_visible(False)
#     ax_waveform.get_yaxis().set_visible(False)
#     ax_waveform.set_title('{0} \n {1}'.format(clip[11:12], os.path.basename(clip)), {'fontsize': 8}, y=1.03)
#
#     librosa.display.specshow(librosa.amplitude_to_db(abs(librosa.stft(audio))), sr=sample_rate, x_axis='time', y_axis='mel', cmap='RdBu_r')
#     ax_spectrogram.get_xaxis().set_visible(False)
#     ax_spectrogram.get_yaxis().set_visible(False)
#
# categories = 5
# clips_shown = 7
# f, axes = plt.subplots(categories, clips_shown, figsize=(clips_shown * 2, categories * 2), sharex=True, sharey=True)
# f.subplots_adjust(hspace=0.35)
# import glob
# file_path = "original"
# cate = [file_path + "/" + x for x in os.listdir(file_path) if os.path.isdir(file_path + "/" + x)]
# labels = []
# audio_name = []
# for idx, folder in enumerate(cate):
#     file_folder = os.listdir(folder)
#     for audio in file_folder[:7]:
#         audio_name.append(folder+"/"+ audio)
#         x = os.path.basename(folder)
#         labels.append((int)(x))
# audio_name = np.array(audio_name).reshape(5,7).tolist()
# for c in range(0, categories):
#     for i in range(0, clips_shown):
#         plot_clip_overview(audio_name[c][i], axes[c, i])
# plt.savefig("wave_spec.pdf")
def add_subplot_axes(ax, position):
    box = ax.get_position()
    position_display = ax.transAxes.transform(position[0:2])
    position_fig = plt.gcf().transFigure.inverted().transform(position_display)
    x = position_fig[0]
    y = position_fig[1]
thisdict ={
  0: "background noise",
  1: "bowhead whale",
  2: "ship signal",
  3: "pilot whale",
  4: "humpback whale"
}

print(thisdict.values())
for x in thisdict.keys():
  print(type(x))
import  matplotlib .pyplot as plt
x=[1,2,3,4,5]
y=[3,6,7,9,2]
# 实例化两个子图(1,2)表示1行2列
fig,ax=plt.subplots(1,2)
ax[0].plot(x,y,label='trend')
ax[1].plot(x,y,color='cyan')
ax[0].set_title(thisdict[1])
ax[1].set_title("{0}".format(thisdict[4]))
plt.show()
import torch
import torch.utils.data as Data
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import librosa
import seaborn as sns
# normalize audio signal to have mean=0 & std=1
def Normalize(waveform):
        return (waveform-waveform.mean()) / waveform.std()

def make_frames(filename,folder,frame_length,overlapping_fraction):
    class_id = os.path.basename(folder)
    filename = folder + '/'+filename
    data, sample_rate = librosa.load(filename, sr=10000)
    data  = Normalize(data)
    stride = int((1 - overlapping_fraction) * frame_length)
    num_frames = int((len(data) - frame_length) / stride) + 1
    temp = np.array([data[i * stride:i * stride + frame_length] for i in range(num_frames)])
    if (len(temp.shape) == 2):
        res = np.zeros(shape=(num_frames, frame_length + 1), dtype=np.float32)
        res[:temp.shape[0], :temp.shape[1]] = temp
        res[:, frame_length] = np.array([class_id] * num_frames)
        return res
def make_frames_folder(folders,frame_length,overlapping_fraction):
    data = []
    for folder in folders:  # folder 0,1,2,3,4
        file_path = 'new_data' + '/' + folder
        files = os.listdir('new_data' + '/' + folder)
        for file in files:
            res = make_frames(file, file_path, frame_length, overlapping_fraction)
            if res is not None:
                data.append(res)
    dataset = data[0]
    for i in range(1, len(data)):
        dataset = np.vstack((dataset, data[i]))
    return dataset

frame_length = 1024
overlapping_fraction = 0.25
def my_data(folders):
    folders = os.listdir(folders)
    dataSet = make_frames_folder(folders, frame_length, overlapping_fraction)
    audio = dataSet[:, 0:frame_length]
    audio = torch.tensor(audio,dtype=torch.float32)
    label = dataSet[:, frame_length]
    label = torch.tensor(label,dtype=torch.int64)
    train_data = Data.TensorDataset(audio, label)
    return train_data
if __name__ == "__main__":
    batch_size = 32
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    dataset = my_data(r'new_data')
    print('=' * 80)
    # ???????????????
    dataset_size = len(dataset)
    print(dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    for step, (b_x, b_y) in enumerate(train_loader):  # torch.Size([64, 1, 28, 28])
        if step > 0:
            break
        # ???????????????batch?????????
        print(b_y)
        batch_x = b_x.numpy()
        batch_y = b_y.numpy()
        dic = {}
        for i in range(len(batch_y)):
            if batch_y[i] in dic:
                continue
            else:
                dic[batch_y[i]] = i
        values = []
        index = []
        for k, v in dic.items():
            values.append(k)
            index.append(v)
        print("????????????label-values", values)
        print("?????????label????????????-index", index)
        # ???????????????batch,???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        # ????????????
        a = [i for i in range(0, batch_size)]
        l = []
        for i in a:
            if i not in index:
                l.append(i)
        print("??????????????????batch?????????lable??????", l)
        x = np.delete(batch_x, l, axis=0)
        print(x.shape)
        plt.figure(figsize=(8, 6))
        # for ii in range(len(values)):
        #     plt.subplot(2, 3, ii + 1)
        #     time_wave = np.arange(0, x.shape[1]) / 10000
        #     plt.plot(time_wave, x[ii, :])
        #     plt.title(values[ii], size=9)
        #     plt.axis("off")
        #     plt.subplots_adjust(wspace=0.05)
        # plt.savefig("plot1.pdf")
        # plt.show()

        # values = sorted(values)
        print(values)
        mapping={0:"ship signal",
                 1:"background noise",
                 2:"humpback whale",
                 3:"pilot whale",
                 4:"bowed whale"}
        print(mapping[1])
        for b in values:
            print(b)
        # print(mapping[b] for b in values)
        plt.grid(True)  # ????????????
        color_list=["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3"]
        # ????????????????????????????????????

        bp = plt.boxplot(x.T,labels=[mapping[b] for b in values],patch_artist=True,notch=True,sym="r+", showmeans=True)
        for i in range(len(values)):
            bp["boxes"][i].set(facecolor=color_list[i])
        # plt.xticks([1,2,3,4,5],["ship signal","background noise","humpback whale","pilot whale","bowhead whale"])
        # plt.xticks([1,2,3,4,5],["ship signal","background noise","humpback whale","pilot whale","bowhead whale"])
        # ??????y_group?????????[[1, 2, 3], [1,2,3,4,5], [8,9]]?????????????????????????????????????????????????????????pos????????????????????????????????????color_list??????????????????N????????????????????????
        # ?????????????????????for???????????????????????????????????????????????????
        # plt.boxplot(x.T, labels=mapping[values], sym="r+", showmeans=True)  # ???????????????
        plt.savefig("xiangxian1.pdf")
        # ??????????????????
        plt.title('Box plot of framed points for the five different types')
        plt.show()
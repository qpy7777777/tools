from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
# 构建数据集
class MyclearDataSet(Dataset):

    def __init__(self,feature_select = None,split=None) :
        self.feature_select = feature_select
        # mfcc_40加载数据
        if self.feature_select == "MFCC_40":
            self.feature = np.load('feature_label/mfcc_40_train_noaug.npy')
            self.label = np.load('feature_label/label_40_train_noaug.npy')

        elif self.feature_select == "mel":
            self.feature = np.load('feature_label/mel_384_train_noaug.npy')
            self.label = np.load('feature_label/label_384_train_noaug.npy')

        elif self.feature_select == "fusion":
            self.feature = np.load('feature_label/fusionFeature_424_train_noaug.npy')
            self.label = np.load('feature_label/label_424_train_noaug.npy')

        elif self.feature_select == "logbank":
            self.feature = np.load('feature_label/logbank_40_train.npy')
            self.label = np.load('feature_label/logbanklabel_40_train.npy')

        elif self.feature_select == "fusion_logbank":
            self.feature = np.load('feature_label/fusionlogbank_424_train.npy')
            self.label = np.load('feature_label/fusionlogbanklabel_424_train.npy')
        elif self.feature_select == "mel_128":
            self.feature = np.load('feature_label/mel_128_train.npy')
            self.label = np.load('feature_label/mellabel_128_train.npy')


        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.feature, self.label, test_size=0.2, random_state=42)
        self.split = split

    def __getitem__(self,index):
        if self.split == 'train':
            x,y = self.x_train[index],self.y_train[index]

        else:
            x,y = self.x_test[index],self.y_test[index]
        return x,y

    def __len__(self):
        if self.split == "train":
            return int(len(self.label)*0.8)
        else:
            return int(len(self.label)*0.2)
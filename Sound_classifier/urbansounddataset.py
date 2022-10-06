import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from torch.utils.data import DataLoader

class UrbanSoundDataset(Dataset):

    def __init__(self,annotation_file,audio_dir,transformation,
                 target_sample_rate,num_samples,device):
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index): #idx的范围是从0到len-1（__len__的返回值）
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        # signal -> (num_channels,samples) -> (2,16000) -> (1,16000)
        # resample
        signal = self._resample_if_necessary(signal,sr)
        # onechannel
        signal = self._mix_down_if_necessary(signal)
        # have same seq length
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        # mel spectrogram
        signal = self.transformation(signal)
        return signal,label

    def _cut_if_necessary(self,signal):
        # signal -> Tensor ->(1,num_samples) ->(1,50000) ->(1,22050)
        if signal.shape[1]  > self.num_samples:
            signal = signal[:,:self.num_samples]
        return signal

    def _right_pad_if_necessary(self,signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0,num_missing_samples) # (0,2) #(1,2) # (1,1,2,2)
            # [1,1,1] -> [1,1,1,0,0] -< [0,1,1,1,0,0]
            # (1,num_samples)
            signal = torch.nn.functional.pad(signal,last_dim_padding)
        return signal

    def _resample_if_necessary(self,signal,sr):
        if sr!= self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr,self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self,signal):
        if signal.shape[0] >1: # (2,10000)
            signal = torch.mean(signal,dim=0,keepdim=True)
        return signal

    def _get_audio_sample_path(self,index):
        fold = f"fold{self.annotations.iloc[index,5]}"
        path = os.path.join(self.audio_dir,fold,self.annotations.iloc[index,0])
        return path

    def _get_audio_sample_label(self,index):
        return self.annotations.iloc[index,6]

if __name__ == "__main__":
    ANNOTATIONS_FILE = "D:/数据集/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "D:/数据集/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    # ms = mel_spectrogram(signal)

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    print(f"There are {len(usd)} samples in the dataset")
    signal, label = usd[0]  # signal ->(1,64,10) ->(channel,melnum,the number of frams)
    a = 1
    # data_loader = DataLoader(usd, batch_size=10, shuffle=True, num_workers=0)
    # for audio, label in data_loader:
    #     print(audio,label)




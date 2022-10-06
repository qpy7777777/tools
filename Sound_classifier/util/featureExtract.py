import glob
import os
import librosa
import numpy as np
from python_speech_features import delta
from python_speech_features import mfcc, delta, logfbank
# fbankfeature

def get_fbank_feature(wavsignal):
    '''
    输入为wav文件数学表示和采样频率，输出为语音的FBANK特征+一阶差分+二阶差分；
    '''
    X, sample_rate = librosa.load(wavsignal, sr=None)
    feat_fbank = logfbank(X, samplerate=sample_rate, nfft=1024,nfilt=40)
    # feat_fbank_d = delta(feat_fbank, 2)
    # feat_fbank_dd = delta(feat_fbank_d, 2)
    # wav_feature = np.column_stack((feat_fbank, feat_fbank_d, feat_fbank_dd))
    # 提取 fbank
    feat_fbank = np.mean(feat_fbank, axis=0)
    return feat_fbank

# mfcc_40特征
def extract_MFCC_40(file_name=None):
    X, sample_rate = librosa.load(file_name,sr=None)
    # 提取MFcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs

# 3D-mel特征
def extract_Mel(file_name = None):
    X, sample_rate = librosa.load(file_name,sr=None)
    melspec = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=1024, hop_length=512, n_mels=128)
    feat_mel_d = delta(melspec, 2)
    feat_mel_dd = delta(feat_mel_d, 2)
    input_feature = np.concatenate((melspec, feat_mel_d, feat_mel_dd))
    input_feature = np.mean(input_feature.T, axis=0)
    return input_feature

def extract_1DMel(file_name = None):
    X, sample_rate = librosa.load(file_name,sr=None)
    melspec = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=1024, hop_length=512, n_mels=128)
    input_feature = np.mean(melspec.T, axis=0)
    return input_feature

# 提取MFCC特征和mel融合特征
def extract_Fusionfeature(file_name=None):
    mfccs = get_fbank_feature(file_name)
    mel = extract_Mel(file_name)
    ext_features = np.hstack([mfccs,mel])
    return ext_features

#解析音频文件

#解析MFCC
def parse_audio_files_mfcc(parent_dir, file_ext="*.wav"):
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    features, labels = np.empty((0, 40)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        # print(label)
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                mfcc_input = extract_MFCC_40(fn)
                ext_features = np.hstack([mfcc_input])
                features = np.vstack([features, ext_features])
                labels = np.append(labels, label)
                # print(np.array(labels, dtype=np.int))
    return np.array(features), np.array(labels, dtype=np.int)
# 解析logbank
def parse_audio_files_logbank(parent_dir, file_ext="*.wav"):
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    features, labels = np.empty((0, 40)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                mfcc_input = get_fbank_feature(fn)
                ext_features = np.hstack([mfcc_input])
                features = np.vstack([features, ext_features])
                labels = np.append(labels, label)
    return np.array(features), np.array(labels, dtype=np.int)
# 解析1D-mel
def parse_audio_files_1Dmel(parent_dir, file_ext="*.wav"):
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    features, labels = np.empty((0, 128)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        # print(label)
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                mfcc_input = extract_1DMel(fn)
                ext_features = np.hstack([mfcc_input])
                features = np.vstack([features, ext_features])
                labels = np.append(labels, label)
                # print(np.array(labels, dtype=np.int))
    return np.array(features), np.array(labels, dtype=np.int)
#解析3D-mel
def parse_audio_files_mel(parent_dir, file_ext="*.wav"):
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    features, labels = np.empty((0, 384)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                mel_input = extract_Mel(fn)
                ext_features = np.hstack([mel_input])
                features = np.vstack([features, ext_features])
                labels = np.append(labels, label)
    return np.array(features), np.array(labels, dtype=np.int)
# 解析融合特征
def parse_audio_files_fusion(parent_dir, file_ext="*.wav"):
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    features, labels = np.empty((0, 424)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                fusio_feature = extract_Fusionfeature(fn)
                ext_features = np.hstack([fusio_feature])
                features = np.vstack([features, ext_features])
                labels = np.append(labels, label)
    return np.array(features), np.array(labels, dtype=np.int)


#文件存储
def npy_save(feature_select):
    #获取特征和标签
    path = "./data_augment"
    # path = "./cut_data"
    # Get features and labels
    if feature_select == "MFCC_40":
        features, labels = parse_audio_files_mfcc(path)
        np.save('feature_label/mfcc_40_train_noaug.npy', features)
        np.save('feature_label/label_40_train_noaug.npy', labels)
    elif feature_select == "mel":
        features, labels = parse_audio_files_mel(path)
        np.save('feature_label/mel_384_train_noaug.npy', features)
        np.save('feature_label/label_384_train_noaug.npy', labels)
    elif feature_select == "fusion":
        features, labels = parse_audio_files_fusion(path)
        np.save('feature_label/fusionFeature_424_train_noaug.npy', features)
        np.save('feature_label/label_424_train_noaug.npy', labels)
    elif feature_select == "logbank":
        features, labels = parse_audio_files_logbank(path)
        np.save('feature_label/logbank_40_train.npy', features)
        np.save('feature_label/logbanklabel_40_train.npy', labels)
    elif feature_select == "fusion_logbank":
        features, labels = parse_audio_files_fusion(path)
        np.save('feature_label/fusionlogbank_424_train.npy', features)
        np.save('feature_label/fusionlogbanklabel_424_train.npy', labels)
    elif feature_select == "mel_128":
        features, labels = parse_audio_files_1Dmel(path)
        np.save('feature_label/mel_128_train.npy', features)
        np.save('feature_label/mellabel_128_train.npy', labels)

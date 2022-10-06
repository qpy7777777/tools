import numpy as np
import librosa
import librosa.display
import glob
import soundfile
import os
# 对原始音频进行数据增强，对每条音频进行加高斯噪（标准差为1 均值为0），SNR=6的高斯白噪
# 时间伸缩，高音修正，波形位移 每条扩充五倍
# 文件保存
def save_file(path):
    if not os.path.exists(path):
        os.mkdirs(path)
        print("文件创建成功")

# 高斯噪声 加性噪声 标准差为1 均值为0
def add_noise(x,w):
    data_nosie = w * np.random.normal(loc=0, scale=1, size=len(x))
    return data_nosie

# 高斯白噪声
def wgn(x, SNR): # wgn是获得原始信号为x,相对于原始信号信噪比是snr dB的高斯噪声
    snr = 10**(SNR/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr # 获取噪声功率
    # 生成标准高斯分布的噪声序列（等于信号长度）,通过转换得到高斯噪声
    return np.random.randn(len(x)) * np.sqrt(npower)

#当参数rate大于1时，Time Stretch执行的是一个在时间维度上压缩的过程，所以视觉上看来是向左偏移了；
def time_stretcheding(audio):
    rate = np.random.uniform(0.8,1.2)
    y_ts = librosa.effects.time_stretch(audio, rate=rate)
    return y_ts
# 经过Pitch Shift转换后的波形明显比原始波形的频率更高一些（n_steps>0），也就是音调变大了
def pitch_shift(audio,fs):
    n_steps = np.random.randint(-5,5)
    y_ps = librosa.effects.pitch_shift(audio,fs,n_steps=n_steps)
    return y_ps
# 波形位移 TimeShift
# # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])-># array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
def time_shift(audio, shift):
    # shift:位移的长度
    y_shift = np.roll(audio, int(shift))
    return y_shift
# 保存所有的信号到wav文件
# 对文件进行保存输出
if __name__ == '__main__':
    new_path  = "../data_augment"
    root = "../cut_data"
    # files = os.listdir(root)
    cate = [root+'/'+x for x in os.listdir(root) if os.path.isdir(root+"/"+x)]
    for idx,folder in enumerate(cate):
        for file in glob.glob(folder+"/*.wav"):
            audio,fs = librosa.load(file,sr=None)
            n = wgn(audio, 6)  # SNR=6
            audio_SNR = audio + n
            audio_noise = audio + add_noise(audio,0.004)
            audio_pitchShift = pitch_shift(audio,fs)
            audio_timeStreching = time_stretcheding(audio)
            audio_timeShift = time_shift(audio,fs)
            audio_new = [audio,audio_SNR,audio_noise,audio_pitchShift,audio_timeStreching,audio_timeShift]
            name = os.path.splitext(os.path.basename(file))[0]
            file_name_0 = name + "_0.wav"
            file_name_1 = "inverse_" + name + "_1.wav"
            file_name_2 = "inverse_" + name + "_2.wav"
            file_name_3 = "inverse_" + name + "_3.wav"
            file_name_4 = "inverse_" + name + "_4.wav"
            file_name_5 = "inverse_" + name + "_5.wav"
            file_name = [file_name_0,file_name_1,file_name_2,file_name_3,file_name_4,file_name_5]
            new_folder = new_path + "/" + os.path.basename(cate[idx])
            print(new_folder)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            for i in range(len(file_name)):
                soundfile.write(os.path.join(new_folder,file_name[i]), audio_new[i],fs)





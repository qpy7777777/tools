import os
import glob
import wavio

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        print("success")

def Cut_file(path):
    folder = [path + "/" + x for x in os.listdir(path) if os.path.isdir(path+ "/" + x)]
    for index, file_dir in enumerate(folder):
        for audio in glob.glob(file_dir+"/*.wav"):
            wav = wavio.read(audio)
            wave_data = wav.data
            str_data = wav.data.reshape(wave_data.shape[0])
            nchannels = wave_data.shape[1]
            sampwidth = wav.sampwidth
            framerate = wav.rate
            # 根据声道数对音频进行转换
            if nchannels > 1:
                wave_data.shape = -1, 2
                wave_data = str_data.T
                temp_data = str_data.T
            else:
                wave_data = str_data.T
                temp_data = str_data.T

            CutFrameNum = framerate * 3
            # print(CutFrameNum)
            Cutnum = wave_data.shape[0] / CutFrameNum  # 音频片段数
            StepNum = int(CutFrameNum)
            # print(Cutnum)
            for j in range(int(Cutnum)):
                FileName = os.path.basename(audio)[0:2] + "-" + str(j + 1) + ".wav"
                out_path = os.path.join(r"../cut_data",os.path.basename(file_dir),FileName)
                temp_dataTemp = temp_data[StepNum * (j):StepNum * (j + 1)]
                create_folder(os.path.dirname(out_path))
                wavio.write(out_path, temp_dataTemp, wav.rate, sampwidth=sampwidth)

if __name__ == "__main__":
    path = "C://Users//lenovo//Desktop//shipEar"
    cutTimeDef = 3  # 裁剪时间 0.5s
    Cut_file(path)
    print("Finishing")

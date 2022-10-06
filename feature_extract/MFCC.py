
import librosa
import numpy as np
from math import floor
from numpy import square, sqrt, mean, abs
from scipy.signal import butter, lfilter, decimate, hilbert
# mfcc_40特征
# 首先提取Mel谱图，得到Mel Bank特征
# Mel-scale, Mel谱长度设置为128。然后对得到的矩阵的列进行压缩，计算每一行的平均值，得到的特征向量
# 返回(128 1)
def extract_MFCC_40(file_name=None):
    X, sample_rate = librosa.load(file_name,sr=None)
    # 提取MFcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs

def subframe_np(sources, fl, fs, scale=False):  # pylint: disable=invalid-name
    """Cuting sources to frames using librosa.
    Args:
        sources (list[np.ndarray,shape==(length,)]): [n] n np array wav datas.
        fl (int): frame length to cut.
        fs (int): frame shift length to cut.
        scale (bool, optional): whether scaler data from [-1,1] to [0,1]. Defaults to False.
    Returns:
        frames (list[np.ndarray,shape==(n_samples, fl)]): frames after cut.
    """
    if scale is True:  # scale from [-1,1] to [0,1]
        for si in sources:  # pylint: disable=invalid-name
            si = si + 1.0  # pylint: disable=invalid-name
            si = si * 0.5  # pylint: disable=invalid-name
    if librosa.__version__ >= '0.7.1':
        frames = [librosa.util.frame(si, fl, fs, axis=0) for si in sources]  # pylint: disable=unexpected-keyword-arg
    else:
        frames = [librosa.util.frame(si, fl, fs).T for si in sources]  # (fl, n_samples)
    return frames

def magnitude_spectrum(source, window, win_length, hop_length, n_fft, center=False,
                       dtype=np.complex64, fix_length=False):  # pylint: disable=too-many-arguments
    """Input 1D frame (must shape (fl,)), return magnitude spectrum shape (1+n_fft/2,t)."""
    if fix_length:
        source = librosa.util.fix_length(source, len(source) + n_fft // 2)
    D = librosa.core.stft(
        source, n_fft=n_fft, hop_length=hop_length, dtype=dtype,
        window=window, win_length=win_length, center=center)
    # amplitude = np.absolute(D)
    # angle = np.angle(D)
    magnitude, phase = librosa.magphase(D)
    feature = magnitude.transpose()
    return feature  # (t, 1+n_fft/2)


def angle_spectrum(source, window, win_length, hop_length, n_fft, center=False,
                   dtype=np.complex64, fix_length=False):  # pylint: disable=too-many-arguments
    """Input 1D frame (must shape (fl,)), return angle spectrum shape (1+n_fft/2,t)."""
    if fix_length:
        source = librosa.util.fix_length(source, len(source) + n_fft // 2)
    D = librosa.core.stft(source, n_fft=n_fft, hop_length=hop_length, dtype=dtype,  # pylint: disable=invalid-name
                          window=window, win_length=win_length, center=center)
    # amplitude = np.absolute(D)
    # angle = np.angle(D)/np.pi
    magnitude, phase = librosa.magphase(D)
    angle = np.angle(phase)/np.pi
    feature = angle.transpose()
    return feature  # (t, 1+n_fft/2)


def real_spectrum(source, window, win_length, hop_length, n_fft, center=False,
                  dtype=np.complex64, fix_length=False):  # pylint: disable=too-many-arguments
    """Input 1D frame (must shape (fl,)), return image spectrum shape (1+n_fft/2,t)."""
    if fix_length:
        source = librosa.util.fix_length(source, len(source) + n_fft // 2)
    D = librosa.core.stft(source, n_fft=n_fft, hop_length=hop_length, dtype=dtype,  # pylint: disable=invalid-name
                          window=window, win_length=win_length, center=center)
    real = np.real(D)
    feature = real.transpose()
    return feature  # (t, 1+n_fft/2)


def image_spectrum(source, window, win_length, hop_length, n_fft, center=False,
                   dtype=np.complex64, fix_length=False):  # pylint: disable=too-many-arguments
    """Input 1D frame (must shape (fl,)), return image spectrum shape (1+n_fft/2,t)."""
    if fix_length:
        source = librosa.util.fix_length(source, len(source) + n_fft // 2)
    D = librosa.core.stft(source, n_fft=n_fft, hop_length=hop_length, dtype=dtype,  # pylint: disable=invalid-name
                          window=window, win_length=win_length, center=center)
    image = np.imag(D)
    feature = image.transpose()
    return feature  # (t, 1+n_fft/2)


def mel_spectrum(source, sr=22050, S=None, n_fft=2048, win_length=None, hop_length=512, window='hann',
                 center=False, pad_mode='reflect', power=2.0, **kwargs):
    """input 1D frame (must shape (fl,)), return mel-scaled spectrogram.
    If a spectrogram input `S` is provided, then it is mapped directly onto
    the mel basis `mel_f` by `mel_f.dot(S)`.
    If a time-series input `y, sr` is provided, then its magnitude spectrogram
    `S` is first computed, and then mapped onto the mel scale by
    `mel_f.dot(S**power)`.  By default, `power=2` operates on a power spectrum.
    Args:
        y : np.ndarray [shape=(n,)] or None; audio time-series
        sr : number > 0 [scalar]; sampling rate of `y`
        S : np.ndarray [shape=(d, t)]; spectrogram
        n_fft : int > 0 [scalar]; length of the FFT window
        hop_length : int > 0 [scalar]; number of samples between successive frames.
        power : float > 0 [scalar]; Exponent for the magnitude melspectrogram.
                                    e.g., 1 for energy, 2 for power, etc.
        kwargs : additional keyword arguments; Mel filter bank parameters. See `librosa.filters.mel` for details.
            n_mels: int > 0 [scalar]; default 128; number of Mel bands to generate
            fmin: float >= 0 [scalar]; lowest frequency (in Hz)
            fmax: float >= 0 [scalar]; highest frequency (in Hz). If None, use fmax = sr / 2.0
    Returns:
        S : np.ndarray [shape=(n_mels, t)]; Mel spectrogram
    """
    feature = librosa.feature.melspectrogram(
        y=source, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=center, pad_mode=pad_mode, power=power, **kwargs)

    return feature


def logmel_spectrum(source=None, sr=22050, S=None, n_fft=2048, win_length=None, hop_length=512, window='hann',
                    center=False, pad_mode='reflect', power=2.0, **kwargs):
    """Input 1d source or power spectrogram S, return Log-Mel Spectrogram."""

    melspec = mel_spectrum(source, sr, S, n_fft, win_length, hop_length, window,
                           center, pad_mode, power, **kwargs)
    feature = librosa.amplitude_to_db(melspec).transpose()

    return feature  # (t, n_mel)


def mfcc(source, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, **kwargs):
    """Input 1D frame (must shape (fl,)), or log-power Mel spectrogram,
       return Mel-frequency cepstral coefficients (MFCCs).
    Args:
        source : np.ndarray, shape=(fl,); input frame
        sr : number > 0 [scalar]; sampling rate of `source`
        S: np.ndarray [shape=(d, t)] or None; log-power Mel spectrogram
        n_mfcc: int > 0 [scalar]; number of MFCCs to return
        dct_type: {1, 2, 3}; Discrete cosine transform (DCT) type. By default, DCT type-2 is used.
        norm: None or ‘ortho’; If dct_type is 2 or 3, setting norm='ortho' uses an ortho-normal DCT basis.
                            Normalization is not supported for dct_type=1.
        lifter: number >= 0; If lifter>0, apply liftering (cepstral filtering) to the MFCCs:
                            M[n, :] <- M[n, :] * (1 + sin(pi * (n + 1) / lifter)) * lifter / 2
                            Setting lifter >= 2 * n_mfcc emphasizes the higher-order coefficients.
                            As lifter increases, the coefficient weighting becomes approximately linear.
        kwargs: additional keyword arguments; Arguments to melspectrogram, if operating on time series input
    Returns:
        M: np.ndarray [shape=(t, n_mfcc)], MFCC sequence
    """
    feature = librosa.feature.mfcc(
        y=source, sr=sr, S=S, n_mfcc=n_mfcc,
        dct_type=dct_type, norm=norm, lifter=lifter, **kwargs).transpose()

    return feature  # (t, n_mfcc)

def feature_extract(feature, **kwargs):

    if feature == 'sample_np':

        features = subframe_np(sources=kwargs['sources'], fl=kwargs['fl'], fs=kwargs['fs'])
    elif feature == 'magspectrum':

        features = magnitude_spectrum(
            source=kwargs['source'], win_length=kwargs['win_length'],
            hop_length=kwargs['hop_length'], n_fft=kwargs['n_fft'],
            window=kwargs['window'], center=kwargs['center'],
            dtype=kwargs['dtype'], fix_length=kwargs['fix_length'])
    elif feature == 'angspectrum':

        features = angle_spectrum(
            source=kwargs['source'], win_length=kwargs['win_length'],
            hop_length=kwargs['hop_length'], n_fft=kwargs['n_fft'],
            window=kwargs['window'], center=kwargs['center'],
            dtype=kwargs['dtype'], fix_length=kwargs['fix_length'])
    elif feature == 'realspectrum':

        features = real_spectrum(
            source=kwargs['source'], win_length=kwargs['win_length'],
            hop_length=kwargs['hop_length'], n_fft=kwargs['n_fft'],
            window=kwargs['window'], center=kwargs['center'],
            dtype=kwargs['dtype'], fix_length=kwargs['fix_length'])
    elif feature == 'imgspectrum':

        features = image_spectrum(
            source=kwargs['source'], win_length=kwargs['win_length'],
            hop_length=kwargs['hop_length'], n_fft=kwargs['n_fft'],
            window=kwargs['window'], center=kwargs['center'],
            dtype=kwargs['dtype'], fix_length=kwargs['fix_length'])
    elif feature == 'logmelspectrum':

        features = logmel_spectrum(**kwargs)
    elif feature == 'mfcc':

        features = mfcc(**kwargs)

    return features

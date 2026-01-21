"""特征提取流水线。"""

import numpy as np
import librosa

from audiofeatures.core.audio_loader import load_audio
from audiofeatures.features.frequency_domain import (
    spectral_bandwidth,
    spectral_centroid,
    spectral_rolloff
)
from audiofeatures.features.spectral import mfcc
from audiofeatures.features.time_domain import zero_crossing_rate


class FeatureExtractor:
    """统一配置的特征提取器。

    Parameters
    ----------
    sr : int, optional
        目标采样率（Hz）。
    n_fft : int, optional
        FFT 点数。
    hop_length : int, optional
        帧移（样本数）。
    n_mels : int, optional
        Mel 滤波器组数量。
    n_mfcc : int, optional
        MFCC 系数数量。

    Attributes
    ----------
    sr : int
        采样率（Hz）。
    n_fft : int
        FFT 点数。
    hop_length : int
        帧移（样本数）。
    n_mels : int
        Mel 滤波器组数量。
    n_mfcc : int
        MFCC 系数数量。
    """

    def __init__(self, sr=22050, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=13):
        """初始化特征提取器。"""
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

    def extract_features(self, signal, feature_types):
        """从信号中提取指定特征。

        Parameters
        ----------
        signal : ndarray
            一维输入信号。
        feature_types : list or tuple
            特征名称列表，支持 ``mfcc``、``spectral_centroid``、
            ``spectral_bandwidth``、``spectral_rolloff``、``zcr``、``rms``、
            ``chroma``、``tonnetz``、``tempogram``。

        Returns
        -------
        dict
            特征字典，键为特征名称，值为特征数组。

        Raises
        ------
        ValueError
            输入非法或特征名称不支持时抛出。
        """
        signal = np.asarray(signal)
        if signal.ndim != 1:
            raise ValueError("signal must be a 1D array")
        if not isinstance(feature_types, (list, tuple)):
            raise ValueError("feature_types must be a list or tuple")

        features = {}
        for feature_type in feature_types:
            if feature_type == "mfcc":
                features[feature_type] = mfcc(
                    signal,
                    sr=self.sr,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels
                )
            elif feature_type == "spectral_centroid":
                features[feature_type] = spectral_centroid(
                    signal,
                    sr=self.sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
            elif feature_type == "spectral_bandwidth":
                features[feature_type] = spectral_bandwidth(
                    signal,
                    sr=self.sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
            elif feature_type == "spectral_rolloff":
                features[feature_type] = spectral_rolloff(
                    signal,
                    sr=self.sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
            elif feature_type == "zcr":
                features[feature_type] = zero_crossing_rate(
                    signal,
                    frame_length=self.n_fft,
                    hop_length=self.hop_length
                )
            elif feature_type == "rms":
                rms = librosa.feature.rms(
                    y=signal,
                    frame_length=self.n_fft,
                    hop_length=self.hop_length
                )
                features[feature_type] = rms.flatten()
            elif feature_type == "chroma":
                chroma = librosa.feature.chroma_stft(
                    y=signal,
                    sr=self.sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                features[feature_type] = chroma
            elif feature_type == "tonnetz":
                harmonic = librosa.effects.harmonic(signal)
                tonnetz = librosa.feature.tonnetz(y=harmonic, sr=self.sr)
                features[feature_type] = tonnetz
            elif feature_type == "tempogram":
                onset_env = librosa.onset.onset_strength(
                    y=signal,
                    sr=self.sr,
                    hop_length=self.hop_length
                )
                tempogram = librosa.feature.tempogram(
                    onset_envelope=onset_env,
                    sr=self.sr,
                    hop_length=self.hop_length
                )
                features[feature_type] = tempogram
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")

        return features

    def extract_from_file(self, file_path, feature_types):
        """从音频文件中提取指定特征。

        Parameters
        ----------
        file_path : str
            音频文件路径。
        feature_types : list or tuple
            特征名称列表。

        Returns
        -------
        dict
            特征字典。
        """
        signal, _ = load_audio(file_path, sr=self.sr, mono=True)
        return self.extract_features(signal, feature_types)

    def extract_all_features(self, signal):
        """提取全部支持的特征类型。

        Parameters
        ----------
        signal : ndarray
            一维输入信号。

        Returns
        -------
        dict
            特征字典。
        """
        feature_types = [
            "mfcc",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "zcr",
            "rms",
            "chroma",
            "tonnetz",
            "tempogram"
        ]
        return self.extract_features(signal, feature_types)

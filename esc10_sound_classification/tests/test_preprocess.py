import pytest
import numpy as np
from src.data.preprocess import reduce_audio_clips, normalize_features

def test_reduce_audio_clips():
    audio_data = [np.random.randn(110250)]  # 5s at 22050 Hz
    fs = 22050
    audio_data_red, time_vector_red = reduce_audio_clips(audio_data, fs, duration=1.25)
    assert len(audio_data_red[0]) == int(fs * 1.25)
    assert len(time_vector_red) == int(fs * 1.25)

def test_normalize_features():
    cwt_mag = np.random.randn(1, 128, 128)
    cwt_phas = np.random.randn(1, 128, 128)
    melspectrogram = np.random.randn(1, 128, 128)
    norm_mag, norm_phas, norm_mel = normalize_features(cwt_mag, cwt_phas, melspectrogram)
    assert norm_mag.shape == cwt_mag.shape
    assert np.abs(norm_mag.mean()) < 1e-5  # StandardScaler mean ~0
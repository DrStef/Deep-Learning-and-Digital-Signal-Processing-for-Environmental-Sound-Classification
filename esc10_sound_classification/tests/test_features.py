import pytest
import numpy as np
from src.features.cwt import compute_cwt
from src.features.mel_spectrogram import compute_mel_spectrogram

def test_compute_cwt():
    audio_data = [np.random.randn(27562)]  # 1.25s at 22050 Hz
    labels = [0]
    scales = np.arange(1, 129)
    dt = 1 / 22050
    mag, phas = compute_cwt(audio_data, labels, scales, dt, step_time=216)
    assert mag.shape == (1, 128, 128)
    assert phas.shape == (1, 128, 128)
    assert np.all(mag >= -40) and np.all(mag <= 5)
    assert np.all(phas >= -3000) and np.all(phas <= 3000)

def test_compute_mel_spectrogram():
    audio_data = [np.random.randn(27562)]  # 1.25s at 22050 Hz
    melspectrogram = compute_mel_spectrogram(audio_data, fs=22050, hop_length=216)
    assert melspectrogram.shape == (1, 128, 128)
    assert np.all(melspectrogram >= -100)  # Reasonable dB range
import numpy as np
import librosa
import time

def compute_mel_spectrogram(audio_data, fs=22050, n_mels=128, n_fft=1024, hop_length=216, trunc_mel=128):
    """
    Compute Mel-spectrograms for audio clips.
    Args:
        audio_data (list): List of audio signals (1.25s each).
        fs (int): Sampling rate (default: 22050 Hz).
        n_mels (int): Number of Mel filters (default: 128).
        n_fft (int): FFT window size (default: 1024).
        hop_length (int): Hop length (default: 216).
        trunc_mel (int): Truncate Mel filters to this number (default: 128).
    Returns:
        melspectrogram (np.array): Mel-spectrograms (N, trunc_mel, time_samples).
    Note:
        Used for ESC-10 dataset, aligned with CWT scalograms for CNN input.
    """
    num_rec=len(audio_data)
    melspectrogram = []

    print('  ')
    print(f'Computing Melspectrograms of all {num_rec} reduced audio clips')
    print('  ')
    cwt_audio_start = time.time()


    for audio in audio_data:
        mel_feat = librosa.feature.melspectrogram(
            y=audio, sr=fs, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
            window='hann', center=True, power=2, pad_mode='constant', n_mels=n_mels
        )
        mel_feat = mel_feat[:trunc_mel, :]
        pwr = librosa.power_to_db(mel_feat, ref=1e-3)
        melspectrogram.append(pwr)

    print('  ')
    print(f'Melspectrograms computed')
    print('  ')
    return np.array(melspectrogram)
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import pywt

def plot_wavelet_properties(wavelet, scales, dt, lev=8):
    """
    Plot real/imaginary parts, magnitude/phase, and frequencies for a wavelet.
    Args:
        wavelet (str): Wavelet type (e.g., 'cmor1.5-1.0').
        scales (np.array): Wavelet scales.
        dt (float): Time step (1/fs).
        lev (int): Decomposition level (default: 8).
    Returns:
        fig (plt.Figure): Wavelet properties plot.
    Note:
        Used to justify wavelet selection for ESC-10 dataset (cmor1.5-1 for harmonics, cgau5 for non-harmonics).
    """
    fig = plt.figure(figsize=(15, 3))
    plt.subplots_adjust(wspace=0.5)

    [psi, x] = pywt.ContinuousWavelet(wavelet).wavefun(lev)
    frequencies = pywt.scale2frequency(wavelet, scales) / dt

    ax = fig.add_subplot(131)
    ax.plot(x, np.real(psi), 'b', label='real')
    ax.plot(x, np.imag(psi), 'r', label='imag')
    ax.legend()
    ax.grid()
    ax.set_title(wavelet)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

    ax = fig.add_subplot(132)
    ax.plot(x, np.abs(psi), 'tab:orange', label='magnitude')
    ax.set_ylabel('Magnitude', color='tab:orange')
    ax.tick_params(axis='y', labelcolor='tab:orange')
    ax.grid()
    ax.set_title(f'Magnitude + Phase {wavelet}')
    ax.set_xlabel('Time')
    ax2 = ax.twinx()
    ax2.plot(x, np.unwrap(np.angle(psi)), 'tab:blue', label='phase')
    ax2.set_ylabel('Rad', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    ax = fig.add_subplot(133)
    ax.semilogy(scales, frequencies, 'k.', label='frequencies')
    ax.grid(which='both')
    ax.set_title(f'Frequencies {wavelet}')
    ax.set_xlabel('Scale')
    ax.set_ylabel('Frequency (Hz)')

    return fig

def plot_cwt_scalograms(scalograms, sub_time, scales, sound_types, is_phase=False):
    """
    Plot CWT magnitude or phase scalograms for sample audio clips.
    Args:
        scalograms (np.array): Scalograms (N, 128, 128).
        sub_time (np.array): Subsampled time vector.
        scales (np.array): Wavelet scales.
        sound_types (list): List of sound type names.
        is_phase (bool): True for phase, False for magnitude (default: False).
    Returns:
        fig (plt.Figure): Scalogram plot.
    """
    contourlevels = np.arange(-3000, 3000, 100) if is_phase else np.arange(-40, 5, 1)
    cmap = 'jet'
    fig = plt.figure(figsize=(20, 17))
    plt.subplots_adjust(hspace=0.4)

    for k in range(len(sound_types)):
        ax = fig.add_subplot(4, 3, 10 if k == 9 else k + 1)
        img = ax.contourf(sub_time, scales, scalograms[k], contourlevels, extend='both', cmap=cmap)
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Phase (rad)' if is_phase else 'Power (dB)')
        ax.set_title(sound_types[k])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Scales')

    return fig

def plot_mel_spectrograms(melspectrogram, sound_types, fs=22050, hop_length=216):
    """
    Plot Mel-spectrograms for sample audio clips.
    Args:
        melspectrogram (np.array): Mel-spectrograms (N, 128, time_samples).
        sound_types (list): List of sound type names.
        fs (int): Sampling rate (default: 22050 Hz).
        hop_length (int): Hop length (default: 216).
    Returns:
        fig (plt.Figure): Mel-spectrogram plot.
    """
    fig = plt.figure(figsize=(20, 17))
    plt.subplots_adjust(hspace=0.4)

    for k in range(len(sound_types)):
        ax = fig.add_subplot(4, 3, 10 if k == 9 else k + 1)
        img = librosa.display.specshow(
            melspectrogram[k], y_axis='linear', hop_length=hop_length, sr=fs,
            x_axis='time', ax=ax, cmap='jet'
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(sound_types[k])
    return fig

def plot_cnn_features(features, indices, sound_types):
    """
    Plot CWT magnitude, phase, and Mel-spectrogram for sample clips.
    Args:
        features (np.array): CNN input features (N, 128, 128, 3).
        indices (list): Indices of clips to plot.
        sound_types (list): List of sound type names.
    Returns:
        fig (plt.Figure): Feature plot.
    """
    fig = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4)
    samples_to_plot = [0, 8, 2]  # Example indices
    titles = ['CWT Magnitude', 'CWT Phase', 'MelSpectrogram']

    for i, sample in enumerate(samples_to_plot):
        for j in range(3):
            ax = fig.add_subplot(3, 3, i * 3 + j + 1)
            plt.imshow(features[indices[sample], ::-1, :, j], cmap='jet', interpolation='none',
                       extent=[0, features.shape[2], 0, features.shape[1]])
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Mels' if j == 2 else 'Scales')
            ax.set_title(f'{sound_types[np.floor(indices[sample]/40).astype("int")]} - {titles[j]}')

    plt.tight_layout()
    return fig

def summarize_diagnostics(history):
    """
    Plot training history for loss and accuracy.
    Args:
        history: Keras history object.
    Returns:
        fig (plt.Figure): Training history plot.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.plot(epochs, loss, 'b', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r', label='Validation Loss')
    ax1.set_title('Cross Entropy Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid()

    ax2.plot(epochs, acc, 'c', label='Training Accuracy')
    ax2.plot(epochs, val_acc, 'm', label='Validation Accuracy')
    ax2.set_title('Classification Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    return fig

def plot_error_analysis(test_X, test_y, predictions, test_indices, features_cwt_CNN):
    """
    Plot features for misclassified samples.
    Args:
        test_X (np.array): Test features (N_test, 128, 128, 3).
        test_y (np.array): Test labels (one-hot).
        predictions (np.array): Predicted probabilities.
        test_indices (np.array): Original indices of test samples.
        features_cwt_CNN (np.array): Full feature set (N, 128, 128, 3).
    Returns:
        fig (plt.Figure): Error analysis plot.
    """
    y_pred = np.argmax(predictions, axis=-1)
    y_true = np.argmax(test_y, axis=-1)
    errors = [i for i in range(len(y_true)) if y_pred[i] != y_true[i]]
    
    if not errors:
        print("No classification errors found.")
        return None

    ind_e = errors[0]
    original_error_index = test_indices[ind_e]
    
    fig = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4)

    titles = ['Wavelet Magnitude', 'Wavelet Phase', 'Mel Spectrogram']
    for j in range(3):
        ax = fig.add_subplot(1, 3, j + 1)
        plt.imshow(test_X[ind_e, ::-1, :, j], cmap='jet', interpolation='none',
                   extent=[0, test_X.shape[2], 0, test_X.shape[1]])
        ax.set_xlabel('Time (instants)')
        ax.set_ylabel('Mels' if j == 2 else 'Scales')
        ax.set_title(f'Pred Error - {titles[j]}')

    plt.tight_layout()
    print(f"Original index of error: {original_error_index}")
    return fig

def plot_class_features(features_cwt_CNN, class_indices, class_name):
    """
    Plot CWT magnitude for all clips in a specific class.
    Args:
        features_cwt_CNN (np.array): Full feature set (N, 128, 128, 3).
        class_indices (list): Indices of clips in the class.
        class_name (str): Class name (e.g., 'SeaWaves').
    Returns:
        fig (plt.Figure): Class feature plot.
    """
    fig = plt.figure(figsize=(14, 24))
    plt.subplots_adjust(hspace=0.8)

    for jj, idx in enumerate(class_indices):
        ax = fig.add_subplot(10, 4, jj + 1)
        plt.imshow(features_cwt_CNN[idx, ::-1, :, 0], cmap='jet', interpolation='none')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Scales')
        ax.set_title(f'{class_name} - Index {idx}')

    return fig
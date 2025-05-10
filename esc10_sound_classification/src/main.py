import numpy as np
from src.data.load_esc10 import load_esc10_data
from src.data.preprocess import reduce_audio_clips, normalize_cwt_features, assemble_features
from src.features.cwt import compute_cwt
from src.features.mel_spectrogram import compute_mel_spectrogram
from src.models.stage2_cnn import prepare_cnn_features, build_cnn_model, train_cnn_model, evaluate_cnn_model

def main():
    """
    Run the full ESC-10 sound classification pipeline.
    - Loads ESC-10 dataset.
    - Preprocesses audio clips to 1.25s.
    - Extracts CWT (magnitude, phase) and Mel-spectrograms.
    - Normalizes and assembles features.
    - Trains and evaluates Stage II CNN.
    Note:
        Stage I CNN (harmonics vs. non-harmonics) is not implemented.
        aT-CWT (proprietary) achieves 100% accuracy but is not included.
    """
    # Load data
    # audio_data, labels, num_classes = load_esc10_data("ESC-10")
    audio_data, labels, num_classes = load_esc10_data("data/raw/ESC-10/")
    print(f"Debug: len(audio_data) = {len(audio_data)}")
    print(f"Debug: len(labels) = {len(labels)}")
    print(f"Debug: num_classes = {num_classes}")

    # Preprocess
    audio_data_red, time_vector_red = reduce_audio_clips(audio_data, fs=22050)
    dt = time_vector_red[1] - time_vector_red[0]
    scales = np.arange(1, 129)
    
    # Feature extraction
    data_cwt_mag, data_cwt_phas = compute_cwt(audio_data_red, labels, scales, dt)
    melspectrogram = compute_mel_spectrogram(audio_data_red, fs=22050)
    
    # Normalize and assemble features
    # norm_cwt_mag, norm_cwt_phas, norm_melspectrogram = normalize_features(data_cwt_mag, data_cwt_phas, melspectrogram)
    # data_cwt_global = assemble_features(norm_cwt_mag, norm_cwt_phas, norm_melspectrogram)

    normalized_data_cwt_mag_vec, normalized_data_cwt_phas_vec, normalized_melspectro_vec = normalize_cwt_features(
    data_cwt_mag, data_cwt_phas, melspectrogram)

    print(f"Debug: size normalized_data_cwt_mag_vec = {np.shape(normalized_data_cwt_mag_vec)}")
    print(f"Debug: size normalized_data_cwt_phas_vec = {np.shape(normalized_data_cwt_phas_vec)}")
    print(f"Debug: size normalized_melspectro_vec = {np.shape(normalized_melspectro_vec)}")   

    data_cwt_global = assemble_features(normalized_data_cwt_mag_vec, normalized_data_cwt_phas_vec, normalized_melspectro_vec)

    # Save features
    np.save('data/processed/normalized_data_cwt_mag_vec.npy', normalized_data_cwt_mag_vec)
    np.save('data/processed/normalized_data_cwt_phas_vec.npy', normalized_data_cwt_phas_vec)
    np.save('data/processed/normalized_melspectro_vec.npy', normalized_melspectro_vec)
    np.save('data/processed/labels.npy', labels)


    # Train and evaluate Stage II CNN
    class_names = [
        'DogBark', 'Rain', 'SeaWaves', 'BabyCry', 'ClockTick',
        'PersonSneeze', 'Helicopter', 'Chainsaw', 'Rooster', 'FireCrackling'
    ]
    train_X, test_X, train_y, test_y, train_indices, test_indices = prepare_cnn_features(data_cwt_global, labels)
    model = build_cnn_model(train_X.shape[1:], num_classes)

    model.summary()


    history = train_cnn_model(model, train_X, train_y, test_X, test_y)
    test_loss, test_acc, report, confusion_fig = evaluate_cnn_model(model, test_X, test_y, class_names)
    
    print(f"Test Accuracy: {test_acc:.4f}\nClassification Report:\n{report}")
    confusion_fig.savefig("docs/figures/CNN_Wavelet_ConfusionMatrix.png")

if __name__ == "__main__":
    main()
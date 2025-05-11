# ESC-10 Sound Classification Pipeline


Classifies sounds like dog barks, rain, and chainsaws using the ESC-10 dataset with wavelets and a CNN, hitting **99% accuracy** with CWT+Melspectrograms and **100% accuracy** with confidential transform aT-CWT (not shared here). 

## Setup     

1. **Clone the repository**:
        ```cmd
        git clone https://github.com/DrStef/Deep-Learning-and-Digital-Signal-Processing-for-Environmental-Sound-Classification.git
        cd Deep-Learning-and-Digital-Signal-Processing-for-Environmental-Sound-Classification/esc10_sound_classification
        ```
2. **Create and activate a virtual environment**:
        ```cmd
        python -m venv venv
        venv\Scripts\activate
        ```
        On Unix/Linux/Mac:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3. **Install requirements**:
        ```cmd
        pip install -r requirements.txt
        ```

4. **Download ESC-10 data**:
        - Get the ESC-10 dataset (~40 MB) from [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50).
        - Place audio files in `data/raw/` (e.g., `data/raw/audio/`).
   

## Running the Pipeline

     Run the full pipeline (loads data, generates features, trains model):
     ```cmd
     python src/main.py
     ```

## Data

     ESC-10 audio (~40 MB) is not included. Download from [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50) and place in `data/raw/`. Processed features (CWT, aT-CWT, Mel spectrograms, ~250 MB) are saved to `data/processed/` by `src/features/cwt.py`, `src/features/at_cwt.py`, or `src/features/mel_spectrogram.py`. Contact DrStef for preprocessed features.

## Folders

     - `src/`: Pipeline code.
       - `data/`: Loading (`load_esc10.py`), preprocessing (`preprocess.py`).
       - `features/`: Feature extraction (`cwt.py`, `at_cwt.py`, `mel_spectrogram.py`).
       - `models/`: CNNs (`stage1_cnn.py`, `stage2_cnn.py`).
       - `utils/`: Metrics, visuals (`metrics.py`, `visualize.py`).
     - `models/`: Trained models.
     - `docs/figures/`: Plots (e.g., `CNN_Wavelet_ConfusionMatrix.png`).
     - `notebooks/`: Analysis notebooks.
     - `tests/`: Unit tests.

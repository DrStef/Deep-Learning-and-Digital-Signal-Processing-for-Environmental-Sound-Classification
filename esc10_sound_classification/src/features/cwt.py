import numpy as np
import pywt
import time

def compute_cwt(audio_data, labels, scales, dt, wavelet_default='cmor1.5-1.0', wavelet_non_harmonics='cgau5', step_time=216):
    """
    Compute CWT magnitude and phase scalograms for audio clips.
    Args:
        audio_data (list): List of audio signals (1.25s each).
        labels (np.array): Class labels (0-9).
        scales (np.array): Wavelet scales (e.g., np.arange(1, 129)).
        dt (float): Time step (1/fs).
        wavelet_default (str): Wavelet for harmonics (default: 'cmor1.5-1.0').
        wavelet_non_harmonics (str): Wavelet for non-harmonics (default: 'cgau5').
        step_time (int): Time subsampling step (default: 216).
    Returns:
        data_cwt_mag (np.array): Magnitude scalograms (N, 128, 128).
        data_cwt_phas (np.array): Phase scalograms (N, 128, 128).
    Note:
        Tailored for ESC-10 dataset, using cmor1.5-1 for harmonics and cgau5 for non-harmonics (rain, sea waves, fire crackling).
    """
    time_red = len(audio_data[0])
    num_rec = len(audio_data)
    fs=22050   # to be modified !!!     fs=1/dt?
    time_vector_red = (1/fs) * np.arange(time_red)
    sub_time = time_vector_red[0:time_red:step_time]
    time_dim = len(sub_time)  # Should be 128
    
    # Initialize arrays
    data_cwt_mag = np.ndarray(shape=(num_rec, len(scales), time_dim))
    data_cwt_phas = np.ndarray(shape=(num_rec, len(scales), time_dim))
    # data_cwt_mag = np.zeros((len(audio_data), len(scales), time_red // step_time))
    # data_cwt_phas = np.zeros((len(audio_data), len(scales), time_red // step_time))

    label_map = {
        '001-Dogbark': 0, '002-Rain': 1, '003-Seawaves': 2, '004-Babycry': 3,
        '005-Clocktick': 4, '006-Personsneeze': 5, '007-Helicopter': 6,
        '008-Chainsaw': 7, '009-Rooster': 8, '010-Firecrackling': 9
    }



    print('  ')
    print(f'Computing Complex Wavelets of all {num_rec} reduced audio clips')
    print('  ')
    cwt_audio_start = time.time()
    
    for ii, (audio, label) in enumerate(zip(audio_data, labels)):
        int_label = label_map[label]  # Convert string to integer

        #print(f" label= {label}" )
        #print(f" labels= {labels}" )

        if ii % 99 == 0:
            print(f"Clip {ii} processed")

        
        wavelet = wavelet_non_harmonics if int_label in [1, 2, 9] else wavelet_default
        coefficients, _ = pywt.cwt(audio, scales, wavelet, dt)
        scal_mag = np.clip(20 * np.log10(np.abs(coefficients) + 1e-12), -40, 5)
        scal_phas = np.clip(np.unwrap(np.angle(coefficients)), -3000, 3000)
        #print(f"Debug: clip {ii}, scal_mag shape = {scal_mag.shape}")
        data_cwt_mag[ii, :, :] = scal_mag[:, 0:time_red:step_time]
        data_cwt_phas[ii, :, :] = scal_phas[:, 0:time_red:step_time]
    
    cwt_audio_end = time.time()
    print(f'CWT computing time (s): {cwt_audio_end - cwt_audio_start:.2f}')
    print('  ')
    return data_cwt_mag, data_cwt_phas





def scale_to_frequency(wavelet, scales, dt):
    """
    Convert wavelet scales to frequencies.
    Args:
        wavelet (str): Wavelet type (e.g., 'cmor1.5-1.0').
        scales (np.array): Wavelet scales.
        dt (float): Time step (1/fs).
    Returns:
        frequencies (np.array): Corresponding frequencies (Hz).
    """
    return pywt.scale2frequency(wavelet, scales) / dt
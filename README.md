# Deep Learning and Digital Signal Processing for Environmental Sound Classification


Automatic environmental sound classification (ESC) based on ESC-50 dataset (and ESC-10 subset) built by Karol Piczak and described in the following article:
"Karol J. Piczak. 2015. ESC: Dataset for Environmental Sound Classification. In Proceedings of the 23rd ACM international conference on Multimedia (MM '15). Association for Computing Machinery, New York, NY, USA, 1015–1018. https://doi.org/10.1145/2733373.2806390"
ESC-50 dataset is available from Dr. Piczak's Github: https://github.com/karoldvl/ESC-50/
The following recent article is a descriptive survey for Environmental sound classification (ESC) detailing datasets, preprocessing techniques, features and classifiers. And their accuracy.
Anam Bansal, Naresh Kumar Garg, "Environmental Sound Classification: A descriptive review of the literature, Intelligent Systems with Applications, Volume 16, 2022, 200115, ISSN 2667-3053, https://doi.org/10.1016/j.iswa.2022.200115.
We develop our own pre-processing techniques for achieving best accuracy results as per Bansal et al .


<ul><ul> The following recent article is a descriptive survey for Environmental sound classification (ESC) detailing datasets, preprocessing techniques, features and classifiers. And their accuracy. </ul></ul>
<ul><ul> Anam Bansal, Naresh Kumar Garg, <b><i> "Environmental Sound Classification: A descriptive review of the literature, 
Intelligent Systems with Applications,</i></b> Volume 16, 2022, 200115, ISSN 2667-3053, https://doi.org/10.1016/j.iswa.2022.200115.  </ul></ul>

<ul><ul> We develop our own pre-processing techniques for achieving best accuracy results as per <i> Bansal et al </i>.  </ul></ul>



ESC-10 Dataset: We are going to use ESC-10 dataset for sound classification. It is a labeled set of 400 environmental recordings (10 classes, 40 clips per class, 5 seconds per clip). It is a subset of the larger ESC-50 dataset

https://github.com/karoldvl/ESC-50/

Each class contains 40 .ogg files. The ESC-10 and ESC-50 datasets have been prearranged into 5 uniformly sized folds so that clips extracted from the same original source recording are always contained in a single fold.

Before we start developing more advanced techniques, we test mel-spectrograms and wavelets. With optimized pre-processing. <br> We will train a Convolution Neural Network model.  <b> At that point, we target an accuracy >= 90 %  </b> 

### Type of sounds/noises:  
<br>
<span style="color:#4169E1"> 
    
- Class = 01-Dogbark, Label = 0
- Class = 02-Rain, Label = 1
- Class = 03-Seawaves, Label = 2
- Class = 04-Babycry, Label = 3
- Class = 05-Clocktick, Label = 4
- Class = 06-Personsneeze, Label = 5
- Class = 07-Helicopter, Label = 6
- Class = 08-Chainsaw, Label = 7
- Class = 09-Rooster, Label = 8
- Class = 10-Firecrackling, Label = 9

 Note: A signal is said to be stationary if its frequency or spectral contents are not changing with respect to time.

    dogbarking, babycry, personsneeze, rooster, involve the vibration and modulation of vocal tract or chords (or nasal "chords"). a bit like speech, and is considered non-stationnary.
    Rain, seawaves are somewhat stationary, rain sounds like a bit like white noise. Let's say pseudo-stationnary because in various recordings other noises are involved at times.
    Helicopter, chainsaw: pseudo-stationary, if the engine r.p.m does not change in a timeframe, the process is stationary. With harmomnics linked to the angular speed and the the number of "pales".
    Firecraking, clock tick: non-stationary, impulsive. The clock tick has the signature of a mechncail vibration radiating sound.

For information about noise from rotating devices: For engine depends on rpm,

By selecting "representative" samples,
    
 There is some confusion:

    in the non-stationary category resultng from vibration of tract, or nasal vibration: dogbark, rooster, person sneeze...
    in the impulsive noise category: clocktick and firecrackling.

We could improve this results, with higher definition mel-spectrogram. Or maybe high definition spectrograms.

    
    
### Mel-Spectrograms and Convolutional Neural Networks

Optimization of mel-spectrogram parameters for best discrimination of sound categories. 

<p align="center"> <img src="Mel-Spectrogram001.png" width="800"  /> </p> 

### Wavelet transform and Convolutional Neural Networks
    
Optimization of wavelet selection and parameters for best discrimination of sound categories. 

<p align="center"> <img src="Mel-Spectrogram001.png" width="800"  /> </p> 

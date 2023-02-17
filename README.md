# Deep Learning and Digital Signal Processing for Environmental Sound Classification

<br>

## Introduction
<br>
Automatic environmental sound classification (ESC) based on ESC-50 dataset (and ESC-10 subset) built by Karol Piczak and described in the following article: <br> 
<br>
"Karol J. Piczak. 2015. <b><i>"ESC: Dataset for Environmental Sound Classification."</i></b> In Proceedings of the 23rd ACM international conference on Multimedia (MM '15). Association for Computing Machinery, New York, NY, USA, 1015–1018. https://doi.org/10.1145/2733373.2806390". <br>
<br>
ESC-50 dataset is available from Dr. Piczak's Github: https://github.com/karoldvl/ESC-50/
The following recent article is a descriptive survey for Environmental sound classification (ESC) detailing datasets, preprocessing techniques, features and classifiers. And their accuracy. <br>
<br>
Anam Bansal, Naresh Kumar Garg, <b><i>"Environmental Sound Classification: A descriptive review of the literature,</i></b> Intelligent Systems with Applications, Volume 16, 2022, 200115, ISSN 2667-3053, https://doi.org/10.1016/j.iswa.2022.200115.  <br>
<br>
We develop our own pre-processing techniques for achieving best accuracy results based on <i>Bansal et al.</i><br>
<span style="color:#4169E1">  <b> At that point, and before we start working on more advanced techniques, we test mel-spectrograms and wavelet transforms. <br> We will train a Convolution Neural Network with spectrograms and scalograms. <br>We target an accuracy >90 %.   </b>    

## Type of sounds/noises   
<br>
 The ESC-10 dataset contains 5 seconds long <b>400 Ogg Vorbis audio clips</b>: sampling frequency: 44.1 kHz, 32- bits float,  and <b>10 classes</b>. <br> 40 audio clips per class.  <br> The 10 Sound/Noise classes are:  <br>  
<br>
  
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

Quick analysis of the type of sound/noise:   
    
- dogbarking, babycry, person sneeze, rooster, involve non-linear vibration and resonance of vocal (or nasal) tract and cords, a bit like speech, and is considered non-stationnary. 
- Rain, sea waves are somewhat stationary, rain sounds a bit like white noise. Pseudo-stationnary because in various audio clips other noises are involved at times. 
- Helicopter, chainsaw: pseudo-stationary. If the engine r.p.m does not change in a timeframe, the process is stationary. With harmonics linked to the engine rpm, number of cylinders, and the number of rotor blades (helicopter).  
- Fire crakling: impulsive noise. But with pseudo-stationary background noise.  
- Clock tick: It depends. Impulsive every second (frequency= 1 Hz). But in some audio clips, there are several "pulsations" in a  1 second time frame. And the ticks have the signature of a non-linear mechanical vibration that radiates sound, with harmonics.
 

## Methodology
 
- In an effort to reduce the size of the problem and computation time, while retaining relevant information, we:  
    - reduce audio sampling frequency from 44.1 kHz to 16 or 24 kHz.     
    - reduce the size of audio clips, to 1.25s, based on signal power considerations. Too many audio clips have occurences of the same sound phenomenon: dog barking, baby crying for example and most of the signal is "silence". 
- Normalize audio signal amplitude to 1. (0 dBFS). 
- Plot mel-spectrograms or Wavelet transforms in the 10 classes. We empirically optimized wavelet selection. And wavelet transform parameters. 
- Reduce the size of scalograms (some details are lost).
- Deep learning with CNN on gray-scale mel-spectrogram scalograms. 
 
We tested three methods: 
 

 
Trained a Convolutional Neural Networks with 32-64-128 256 neurons Hidden Layers. Parameters are detailed in the various notebooks. 
 
 
## Results Synthesis  
 
 
| Method | Accuracy |
| ---     |  ---    |
| 256x256 Mel-spectrograms |      92.5 %   |
| 128x128 Complex Wavelet Transforms Scalograms Magnitude + Phase |     92.5 %  |
 | 128x128 Fusion Complex Wavelet Transform + Mel-Spectrograms | <b>97.5 %</b>  |
 
Best Result: 

<div align="center"> 
 
|<p align="center">   <img src="https://github.com/DrStef/Deep-Learning-and-Digital-Signal-Processing-for-Environmental-Sound-Classification/blob/main/CNN_Fusion_ClassificationReport_98pc.png"  width="350"  /> </p>    |  <p align="center"> <img src="https://github.com/DrStef/Deep-Learning-and-Digital-Signal-Processing-for-Environmental-Sound-Classification/blob/main/CNN_Fusion_ConfusionMatrix_98pc.png" width="300"  /> </p> |  
| ---       | ---                          |   
|<p align="center"> <i> Classification report </i> </p>  |  <p align="center"> <i> Confusion matrix </i> </p>       |  

 </div>
 
 <br>
 <br>
 

## Jupyter Notebooks  

All Jupyter Notebooks sahre the same structure. They are identical except for wavelet transforms or mel-spectrogram transforms. 
 
###  Part III: Fusion: Complex Wavelet Transforms + Mel-Spectrograms and CNN 
 
 
<b>Applying different wavelets to each type of sound significantly improves CNN Deep Learning accuracy. ~ 98%. </b>
<br>
 
 
###  Part II: Complex Wavelet Transform and Convolutional Neural Networks (CNN)
    
Optimization of wavelet selection and parameters for best discrimination of sound categories. <br>
Wavelet selection: the difficulty here is the selection of the right wavelet suited to the full range of noise types: pseudo-stationary, non-stationary, transient/impulsive. <br>
<b>Combining Mel-Spectrograms with Complex Wavelets Transforms enhances accuracy with features that are difficult to discriminate.  Accuracy. ~ 92%. </b>
<br>
<p align="center"> <img src="Wavelets_transform3_002.png" width="800"  /> </p> 
 
 
   
###  Part I: Mel-Spectrograms and Convolutional Neural Networks (CNN)

Reduction of audio clips size and optimization of mel-spectrogram parameters for best discrimination of sound categories. ~92%

<p align="center"> <img src="https://github.com/DrStef/Deep-Learning-and-Digital-Signal-Processing-for-Environmental-Sound-Classification/blob/main/Melspectrogram_91pcA.png" width="800"  /> </p> 
  




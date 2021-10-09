# trimodal_person_verification
This repository contains the code, pretrained models and preprocessed dataset featured in "A Study of Multimodal Person Verification Using Audio-Visual-Thermal Data".

Person verification is the general task of verifying person’s identity  using  various  biometric  characteristics. We study an approach to multimodal person verification using audio, visual, and thermal modalities. In particular, we implemented unimodal, bimodal, and trimodal verification systems using the state-of-the-art deep learning architectures and compared their performance under clean and noisy conditions.
### Dependencies
```
pip install -r requirements.txt
```
### Dataset
In this work, we utilized the SpeakingFaces dataset to train, validate, and test the person verification systems. SpeakingFaces is a publicly available multimodal dataset comprised of audio, visual, and thermal data streams. The preprocessed data used for our experiments can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1AcK6ETnmZzNXHi5qGugRKXz-wnebDX0-?usp=sharing). 

The *metadata* subdirectory contains lists prepared for the train, validation, and test sets following the format of VoxCeleb. In particular, the train list contains the paths to the recordings and the corresponding subject identifiers. The validation and test lists consist of randomly generated positive and negative pairs. For each subject, the same number of positive and negative pairs were selected. In total, the numbers of pairs in the validation and test sets are 38,000 and 46,200, respectively.
### Training examples

### Evaluating pretrained models

### Noisy data

### Citation



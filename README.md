# trimodal_person_verification
This repository contains the code, pretrained models and preprocessed dataset featured in "A Study of Multimodal Person Verification Using Audio-Visual-Thermal Data".

Person verification is the general task of verifying person’s identity  using  various  biometric  characteristics. We study an approach to multimodal person verification using audio, visual, and thermal modalities. In particular, we implemented unimodal, bimodal, and trimodal verification systems using the state-of-the-art deep learning architectures and compared their performance under clean and noisy conditions.
### Dependencies
```
pip install -r requirements.txt
```
### Dataset
In this work, we utilized the SpeakingFaces dataset to train, validate, and test the person verification systems. SpeakingFaces is a publicly available multimodal dataset comprised of audio, visual, and thermal data streams. The preprocessed data used for our experiments can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1AcK6ETnmZzNXHi5qGugRKXz-wnebDX0-?usp=sharing). 

The *data* directory contains the compressed version of the preprocessed data used for the reported experiments. For each utterance, only the first frame (visual and thermal) is selected. The train set is split into 5 parts that should be extracted into the same location.

The *data/metadata* subdirectory contains lists prepared for the train, validation, and test sets following the format of VoxCeleb. In particular, the train list contains the paths to the recordings and the corresponding subject identifiers. The validation and test lists consist of randomly generated positive and negative pairs. For each subject, the same number of positive and negative pairs were selected. In total, the numbers of pairs in the validation and test sets are 38,000 and 46,200, respectively.

Note, to run noisy training and evaluation, you should first download the [MUSAN dataset](http://www.openslr.org/17/).

See *trainSpeakerNet.py* for details on where the data should be stored.

### Training examples : clean data
*Unimodal models*
```
python trainSpeakerNet.py --model ResNetSE34Multi --modality wav --log_input True --trainfunc angleproto --max_epoch 1500 --batch_size 100 --nPerSpeaker 9 --max_frames 200 --eval_frames 200 --weight_decay 0.01 --seed 1 --save_path exps/wav/exp1 
```

```
python trainSpeakerNet.py --model ResNetSE34Multi --modality rgb --log_input True --trainfunc angleproto --max_epoch 600 --batch_size 100 --nPerSpeaker 9 --max_frames 200 --eval_frames 200 --weight_decay 0.01 --seed 1 --save_path exps/rgb/exp1 
```

```
python trainSpeakerNet.py --model ResNetSE34Multi --modality thr --log_input True --trainfunc angleproto --max_epoch 600 --batch_size 100 --nPerSpeaker 9 --max_frames 200 --eval_frames 200 --weight_decay 0.01 --seed 1 --save_path exps/thr/exp1 
```
*Multimodal models*
```
python trainSpeakerNet.py --model ResNetSE34Multi --modality wavrgb --log_input True --trainfunc angleproto --max_epoch 600 --batch_size 100 --nPerSpeaker 9 --max_frames 200 --eval_frames 200 --weight_decay 0.01 --seed 1 --save_path exps/wavrgb/exp1 
```

```
python trainSpeakerNet.py --model ResNetSE34Multi --modality wavrgbthr --log_input True --trainfunc angleproto --max_epoch 600 --batch_size 100 --nPerSpeaker 9 --max_frames 200 --eval_frames 200 --weight_decay 0.1 --seed 1 --save_path exps/wavrgb/exp1 
```

### Training examples : noisy data
*Unimodal models*
```
python trainSpeakerNet.py --model ResNetSE34Multi --modality wav --noisy_train True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --max_epoch 1500 --batch_size 100 --nPerSpeaker 9 --max_frames 200 --eval_frames 200 --weight_decay 0.001 --seed 1 --save_path exps/wav/exp2
```

```
python trainSpeakerNet.py --model ResNetSE34Multi --modality rgb --noisy_train True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --max_epoch 600 --batch_size 100 --nPerSpeaker 9 --max_frames 200 --eval_frames 200 --weight_decay 0.01 --seed 1 --save_path exps/rgb/exp2 
```

```
python trainSpeakerNet.py --model ResNetSE34Multi --modality thr --noisy_train True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --max_epoch 600 --batch_size 100 --nPerSpeaker 9 --max_frames 200 --eval_frames 200 --weight_decay 0.01 --seed 1 --save_path exps/thr/exp2 
```
*Multimodal models*
```
python trainSpeakerNet.py --model ResNetSE34Multi --modality wavrgb --noisy_train True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --max_epoch 600 --batch_size 100 --nPerSpeaker 9 --max_frames 200 --eval_frames 200 --weight_decay 0.01 --seed 1 --save_path exps/wavrgb/exp2 
```

```
python trainSpeakerNet.py --model ResNetSE34Multi --modality wavrgbthr --noisy_train True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --max_epoch 600 --batch_size 100 --nPerSpeaker 9 --max_frames 200 --eval_frames 200 --weight_decay 0.1 --seed 1 --save_path exps/wavrgb/exp2 
```

### Evaluating pretrained models: clean test data
*Unimodal models*
```
python trainSpeakerNet.py --model ResNetSE34Multi --modality wav --eval True --valid_model True --test_path data/test --test_list data/metadata/test_list.txt --log_input True --trainfunc angleproto --eval_frames 200 --save_path exps/wav/exp1 
```

```
python trainSpeakerNet.py --model ResNetSE34Multi --modality rgb --eval True --valid_model True --test_path data/test --test_list data/metadata/test_list.txt   --log_input True --trainfunc angleproto --eval_frames 200 --save_path exps/rgb/exp1 
```

```
python trainSpeakerNet.py --model ResNetSE34Multi --modality thr --eval True --valid_model True --test_path data/test --test_list data/metadata/test_list.txt   --log_input True --trainfunc angleproto --eval_frames 200 --save_path exps/thr/exp1 
```
*Multimodal models*
```
python trainSpeakerNet.py --model ResNetSE34Multi --modality wavrgb  --eval True --valid_model True --test_path data/test --test_list data/metadata/test_list.txt   --log_input True  --trainfunc angleproto --eval_frames 200 --save_path exps/wavrgb/exp1 
```

```
python trainSpeakerNet.py --model ResNetSE34Multi --modality wavrgbthr --eval True --valid_model True --test_path data/test --test_list data/metadata/test_list.txt   --log_input True  --trainfunc angleproto --eval_frames 200 --save_path exps/wavrgb/exp1 
```
### Evaluating pretrained models: noisy test data

*Unimodal models*

```
python revalidate.py --model ResNetSE34Multi --modality wav --noisy_eval True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --eval_frames 200 --save_path exps/wav/exp2

python revalidate.py --model ResNetSE34Multi --modality wav --eval True --valid_model True --test_path data/test --test_list data/metadata/test_list.txt    --noisy_eval True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --eval_frames 200 --save_path exps/wav/exp2
```

```
python revalidate.py --model ResNetSE34Multi --modality rgb --noisy_eval True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --eval_frames 200 --save_path exps/rgb/exp2

python revalidate.py --model ResNetSE34Multi --modality rgb --eval True --valid_model True --test_path data/test --test_list data/metadata/test_list.txt    --noisy_eval True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --eval_frames 200 --save_path exps/rgb/exp2 
```

```
python revalidate.py --model ResNetSE34Multi --modality thr --noisy_eval True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --eval_frames 200 --save_path exps/thr/exp2

python revalidate.py --model ResNetSE34Multi --modality thr --eval True --valid_model True --test_path data/test --test_list data/metadata/test_list.txt    --noisy_eval True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --eval_frames 200 --save_path exps/thr/exp2 
```

*Multimodal models*
```
python revalidate.py --model ResNetSE34Multi --modality wavrgb --noisy_eval True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --eval_frames 200 --save_path exps/wavrgb/exp2

python revalidate.py --model ResNetSE34Multi --modality wavrgb --eval True --valid_model True --test_path data/test --test_list data/metadata/test_list.txt    --noisy_eval True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --eval_frames 200 --save_path exps/wavrgb/exp2 
```

```
python revalidate.py --model ResNetSE34Multi --modality wavrgbthr --noisy_eval True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --eval_frames 200 --save_path exps/wavrgbthr/exp2

python revalidate.py --model ResNetSE34Multi --modality wavrgbthr --eval True --valid_model True --test_path data/test --test_list data/metadata/test_list.txt    --noisy_eval True --p_noise 0.3 --snr 8 --log_input True --trainfunc angleproto --eval_frames 200 --save_path exps/wavrgb/exp2 
```




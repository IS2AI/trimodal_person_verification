# trimodal_person_verification
This repository contains the code, and preprocessed dataset featured in "A Study of Multimodal Person Verification Using Audio-Visual-Thermal Data" and "Multimodal Person Verification with Thermal Data Augmentation"

### Dependencies
```
pip install -r requirements.txt
```
### Dataset

The preprocessed data used for our experiments can be downloaded from [Google Drive](https://drive.google.com/drive/folders/13ZR2IxHzsbgE72_mC-ltUTdqKjPAEtEy?usp=sharing).

The *data* directory contains the compressed version of the preprocessed data used for the reported experiments. For each utterance, only the first frame (visual and thermal) is selected. The train set is split into 5 parts that should be extracted into the same location. 


The *data/metadata* contains lists prepared for the train, validation, and test sets for the experiments featured in both papers:
1) *train_list.txt* contains the paths to the recordings and the corresponding subject identifiers present in SpeakingFaces. 
2) The *valid_list.txt* and *test_list.txt* consist of randomly generated positive and negative pairs taken from the validation and test splits of SpeakingFaces, respectively. For each subject, the same number of positive and negative pairs were selected. In total, the numbers of pairs in the validation and test sets are 38,000 and 46,200, respectively.
3) *train_list_1_percent.txt*, *train_list_5_percent.txt*, *train_list_100_percent.txt* were designed to combine the train splits of SpeakingFaces and VoxCeleb datasets. 
*train_list_1_percent.txt* - the total number of subjects was increased from 100 to 160 by adding a random 1% of training subjects from VoxCeleb.  *train_list_5_percent.txt* - 4% of new random subjects from the VoxCeleb2 development split were added to the existing 1% in *train_list_1_percent.txt*.
*train_list_100_percent.txt* - all data from the VoxCeleb2 development split were combined with SpeakingFaces

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




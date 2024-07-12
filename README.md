<div align="center">

# Frequency Masking for Universal DeepFake Detection
[Paper](https://arxiv.org/abs/2401.06506)
</div>

<p align="center">
  <img src="https://github.com/chandlerbing65nm/FakeDetection/assets/62779617/d0564928-96ea-48ff-b2c9-93743340128b" width="350" height="350">
</p>

>We study universal deepfake detection.
>Our  goal is to detect synthetic images from a range of generative AI approaches, particularly emerging ones which are unseen during training of the deepfake detector.
>Universal deepfake detection requires outstanding generalization capability.
>Motivated by recently proposed masked image modeling which has demonstrated excellent generalization in self-supervised pre-training, we make the first attempt to explore masked image modeling for universal deepfake detection. We study spatial and frequency domain masking in training deepfake detector.
>Based on empirical analysis, we propose a novel deepfake detector via frequency masking.
>Our focus on frequency domain is different from most spatial domain  detection. Comparative analyses reveal substantial performance gains over existing methods. 

## PLEASE READ DOCUMENTATION BELOW

## &#9733; Datasets
Follow strictly the naming and structure of folders below:

### [Wang_CVPR2020](https://github.com/PeterWang512/CNNDetection/tree/195892d93fc3f26599f93d8d9e1ca995991da2ee)
- training/validation: [link](https://drive.google.com/file/d/1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh/view)
- testing: [link](https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view)
```
Wang_CVPR2020/
├── testing/
│   ├── crn/
│   │   ├── 1_fake/
│   │   └── 0_real/
│   └── ...
├── training/
│   ├── car/
│   │   ├── 1_fake/
│   │   └── 0_real/
│   └── ...
└── validation/
    ├── car/
    │   ├── 1_fake/
    │   └── 0_real/
    └── ...
```

### [Ojha_CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect)
- download: [link](https://drive.google.com/file/d/1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t/view)
```
Ojha_CVPR2023/
├── guided/
│   ├── 1_fake/
│   └── 0_real/
├── ldm_200_cfg/
│   ├── 1_fake/
│   └── 0_real/
├── ldm_100/
│   ├── 1_fake/
│   └── 0_real/
└── ...
```

Make sure to change the path based on where you saved the data: [link1](https://github.com/chandlerbing65nm/FakeDetection/blob/9699fa5137420ffc611885fc79479a99cd313438/train.py#L113C1-L120C97) [link2](https://github.com/chandlerbing65nm/FakeDetection/blob/9699fa5137420ffc611885fc79479a99cd313438/test.py#L114C1-L145C47)

## &#9733; Model Weights

You can download the model weights [here](https://drive.google.com/drive/folders/1ePTY4x2qvD7AVlNJXFLozFbUF6Y0_hET?usp=sharing). Put the files under the repository and don't change the name or anything!

File structure should strictly be like this:
```
FakeImageDetection/checkpoints/
├── mask_0/
│   ├── rn50ft.pth (Wang et al.)
│   ├── rn50_modft.pth (Gragnaniello et al.)
│   ├── clipft.pth (Ojha et al.)
│   └── ...
├── mask_15/
│   ├── rn50ft_midspectralmask.pth
│   ├── rn50ft_lowspectralmask.pth
│   ├── rn50ft_highspectralmask.pth
│   ├── rn50ft_pixelmask.pth
│   ├── rn50ft_patchmask.pth
│   ├── rn50ft_spectralmask.pth (Wang et al. + Ours)
│   ├── rn50_modft_spectralmask.pth (Gragnaniello et al. + Ours)
│   ├── clipft_spectralmask.pth (Ojha et al. + Ours)
│   └── ...
└── ...
```

## &#9733; Testing Script (test.py)

### Description
The script `test.py` is designed for evaluating trained models on multiple datasets. The script leverages metrics such as Average Precision, Accuracy, and Area Under the Curve (AUC) for evaluation.

### Basic Command

```bash
python -m torch.distributed.launch --nproc_per_node=GPU_NUM test.py -- [options]
```
Command-Line Options
```bash
--model_name : Type of the model. Choices include various ResNet and ViT variants.
--mask_type  : Type of mask generator for data augmentation. Choices include 'patch', 'spectral', etc.
--pretrained : Use pretrained model.
--ratio      : Masking ratio for data augmentation.
--band       : Frequency band to randomly mask.
--batch_size : Batch size for evaluation. Default is 64.
--data_type  : Type of dataset for evaluation. Choices are 'Wang_CVPR20' and 'Ojha_CVPR23'.
--device     : Device to use for evaluation (default: auto-detect).
```

### Bash Command
Edit testing bash script `test.sh`:

```bash
#!/bin/bash

# Define the arguments for your test script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
DATA_TYPE="Wang_CVPR20"  # Wang_CVPR20 or Ojha_CVPR23
MODEL_NAME="clip" # clip, RN50_mod or RN50
MASK_TYPE="nomask" # spectral, pixel, patch or nomask
BAND="all" # all, low, mid, high
RATIO=15
BATCH_SIZE=64

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs

echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Run the test command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU test.py \
  -- \
  --data_type $DATA_TYPE \
  --pretrained \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE \
  --other_model
```

Now, use this to run testing:
```bash
bash test.sh "0" # gpu id/s to use
``` 

### Results
You can find the results in this structure:

```bash
results/
├── mask_15/
│   ├── ojha_cvpr23/
│   │   ├── rn50ft_spectralmask.txt
│   │   └── ...
│   └── wang_cvpr20/
│       ├── rn50ft_spectralmask.txt
│       └── ...
└── ...
```


## &#9733; Training Script (train.py)

### Description

This script `(train.py)` is designed for distributed training and evaluation of models. The script is highly configurable through command-line arguments and provides advanced features such as `WandB` integration, early stopping, and various masking options for data augmentation.

### Basic Command

To run the script in a distributed environment:

```bash
python -m torch.distributed.launch --nproc_per_node=GPU_NUM train.py -- [options]

```

Command-Line Options

```bash
--local_rank     : Local rank for distributed training. 
--num_epochs     : Number of epochs for training. 
--model_name     : Type of the model. Choices include various ResNet and ViT variants.
--wandb_online   : Run WandB in online mode. Default is offline.
--project_name   : Name of the WandB project.
--wandb_run_id   : WandB run ID.
--resume_train   : Resume training from last or best epoch.
--pretrained     : Use pretrained model.
--early_stop     : Enable early stopping.
--mask_type      : Type of mask generator for data augmentation. Choices include 'patch', 'spectral', etc.
--batch_size     : Batch size for training. Default is 64.
--ratio          : Masking ratio for data augmentation.
--band           : Frequency band to randomly mask.
```

### Bash Command
Edit training bash script `train.sh`:

```bash
#!/bin/bash

# Get the current date
current_date=$(date)

# Print the current date
echo "The current date is: $current_date"

# Define the arguments for your training script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
NUM_EPOCHS=10000
PROJECT_NAME="Frequency-Masking"
MODEL_NAME="clip" # RN50_mod, RN50, clip
MASK_TYPE="spectral" # nomask, spectral, pixel, patch
BAND="all" # all, low, mid, high
RATIO=15
BATCH_SIZE=128
WANDB_ID="2w0btkas"
RESUME="from_last" # from_last or from_best

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs

echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Run the distributed training command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train.py \
  -- \
  --num_epochs $NUM_EPOCHS \
  --project_name $PROJECT_NAME \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE \
  --early_stop \
  --pretrained \
  # --resume_train $RESUME \
  # --debug \
  # --wandb_online \
  # --wandb_run_id $WANDB_ID \
```

Now, use this to run training:
```bash
bash train.sh "0,1,2,4" # gpu ids to use
```

Important:
- When starting the training (from 1st epoch), please comment out  `--resume_train $RESUME \` and `--debug \`. And if you don't want to use `wandb` logging, comment out `--wandb_online \` and `--wandb_run_id $WANDB_ID \`. In short, just comment out the last three lines in the bash script.
- If you notice that the training process stalls during an epoch (e.g., epoch 20+ or 30+), please interrupt it by pressing ctrl+c. The bash script is configured to resume training from the last saved epoch (if you uncomment `--resume_train $RESUME \`).

## &#9733; License

This project is licensed under the [Apache License](LICENSE).

## &#9733; Citation

If you use this code in your research, please consider citing it. Below is the BibTeX entry for citation:

```bibtex
@misc{doloriel2024frequency,
      title={Frequency Masking for Universal Deepfake Detection}, 
      author={Chandler Timm Doloriel and Ngai-Man Cheung},
      year={2024},
      eprint={2401.06506},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

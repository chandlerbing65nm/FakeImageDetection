<div align="center">

# Frequency Masking for Universal DeepFake Detection

</div>

<div align="justify">

As artificial intelligence increasingly facilitates the creation of convincing synthetic images, the need for effective universal deepfake detection has never been more critical. Existing approaches, primarily based on CNNs and Transformers, often yield suboptimal results due to an overemphasis on irrelevant features. In this work, we shift the paradigm by focusing on the frequency domain through the use of Fast Fourier Transform (FFT). We introduce a frequency-based masking technique explicitly designed for universal deepfake detection. Our method strategically targets specific frequency bands, capturing forensically important features often overlooked by conventional approaches. Comparative analyses reveal substantial performance gains over existing methods. This investigation marks a significant insights in the field of universal deepfake detection, supported by a comprehensive suite of analytical insights and experimental evidence. 

</div>

<p align="center">
  <img src="https://github.com/chandlerbing65nm/FakeDetection/assets/62779617/d0564928-96ea-48ff-b2c9-93743340128b" width="500" height="500">
</p>

## &#9733; Datasets
### [Wang_CVPR2020](https://github.com/PeterWang512/CNNDetection/tree/195892d93fc3f26599f93d8d9e1ca995991da2ee)
Follow strictly the naming and structure of folders below:
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

## &#9733; Pretrained Models

You can download the pretrained weights [here](https://drive.google.com/drive/folders/1ePTY4x2qvD7AVlNJXFLozFbUF6Y0_hET?usp=sharing). Put the files under the repository and don't change the name or anything!

File structure should strictly be like this:
```
checkpoints/
├── mask_15/
│   ├── rn50ft_midspectralmask.pth
│   ├── rn50ft_lowspectralmask.pth
│   ├── rn50ft_highspectralmask.pth
│   ├── rn50ft_spatialmask.pth (pixel masking in paper)
│   ├── rn50ft_patchmask.pth
│   ├── rn50ft_spectralmask.pth (Wang et al. + Ours)
│   └── rn50_modft_spectralmask.pth (Gragnaniello et al. + Ours)
│   └── ...
└── ...
```

## &#9733; Testing Script (test.py)

### Description
The script `test.py` is designed for evaluating trained models on multiple datasets. The script leverages metrics such as Average Precision, Accuracy, and Area Under the Curve (AUC) for evaluation.

### Basic Command

```bash
python test.py [options]
```
Command-Line Options
```bash
--model_name : Type of the model. Choices include various ResNet and ViT variants.
--mask_type  : Type of mask generator for data augmentation. Choices include 'patch', 'spectral', etc.
--pretrained : Use pretrained model.
--ratio      : Masking ratio for data augmentation.
--batch_size : Batch size for evaluation. Default is 64.
--data_type  : Type of dataset for evaluation. Choices are 'Wang_CVPR20' and 'Ojha_CVPR23'.
--device     : Device to use for evaluation (default: auto-detect).
```

### Bash Command
Edit testing bash script:

```bash
#!/bin/bash

# Define the arguments for your test script
DATA_TYPE="Wang_CVPR20"  # Wang_CVPR20 or Ojha_CVPR23
MODEL_NAME="RN50"
MASK_TYPE="spectral"
RATIO=15
BATCH_SIZE=64
DEVICE="cuda:0"

# Run the test command
python test.py \
  --data_type $DATA_TYPE \
  --pretrained \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE \
  --device $DEVICE
```
Now, use this to run testing:
```bash
bash test.sh
```

### Results
You can find the results in this structure:

```
results/
├── mask_15/
└── ├── ojha_cvpr23/
    └── ├── rn50ft_spectralmask.txt
        └── ...
    ├── wang_cvpr20/
    └── ├── rn50ft_spectralmask.txt
        └── ...
```


## &#9733; Training Script (train.py)

### Description

This script `(train.py)` is designed for distributed training and evaluation of various Deep Learning models including ResNet variants. The script is highly configurable through command-line arguments and provides advanced features such as `WandB` integration, early stopping, and various masking options for data augmentation.

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
```

### Bash Command
Edit training bash script:

```bash
#!/bin/bash

# Define the arguments for your training script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
NUM_EPOCHS=10000
PROJECT_NAME="Frequency-Masking"
MODEL_NAME="RN50_mod" # RN50_mod, RN50
MASK_TYPE="spectral"
RATIO=15
BATCH_SIZE=16
WANDB_ID="qvlglly2"
RESUME="from_last" # from_last or from_best

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=$GPUs

echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Run the distributed training command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train.py \
  -- \
  --num_epochs $NUM_EPOCHS \
  --project_name $PROJECT_NAME \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE \
  --early_stop \
  --pretrained \
  --wandb_online \
  --wandb_run_id $WANDB_ID \
  --resume_train $RESUME \
```

Now, use this to run training:
```bash
bash train.sh "0,1,2,4" # gpu ids to use
```

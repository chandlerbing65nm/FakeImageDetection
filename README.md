<div align="center">

# Frequency Masking for Universal DeepFake Detection
[Paper](https://arxiv.org/abs/2401.06506)
</div>

<!-- <p align="center">
  <img src="https://github.com/chandlerbing65nm/FakeDetection/assets/62779617/d0564928-96ea-48ff-b2c9-93743340128b" width="350" height="350">
</p> -->

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

Paths to datasets are currently hardcoded in `train.py` and `test.py` under `/mnt/SCRATCH/chadolor/Datasets/...`. Please update these paths to match your environment before running.

## &#9733; Model Weights

You can download sample weights [here](https://drive.google.com/drive/folders/1ePTY4x2qvD7AVlNJXFLozFbUF6Y0_hET?usp=sharing).

By default, both training and testing use an absolute checkpoints directory under:
```
/mnt/SCRATCH/chadolor/Datasets/Projects/FakeImageDetector/checkpoints/
```

Within that directory, checkpoints are grouped by mask ratio (percent):
```
/mnt/SCRATCH/chadolor/Datasets/Projects/FakeImageDetector/checkpoints/
├── mask_0/
│   ├── rn50ft.pth
│   ├── rn50_modft.pth
│   ├── clipft.pth
│   └── ...
├── mask_15/
│   ├── rn50ft_lowfouriermask.pth
│   ├── rn50ft_midfouriermask.pth
│   ├── rn50ft_highfouriermask.pth
│   ├── rn50ft_pixelmask.pth
│   ├── rn50ft_patchmask.pth
│   ├── rn50ft_fouriermask_rotate_chr.pth       # example: frequency + combine_aug + channel suffix
│   └── ...
└── ...
```
Notes:
- Filename pattern encodes model, band, mask_type, optional `combine_aug` (e.g., `_rotate`, `_translate`, `_rotate_translate`), and optional channel suffix (e.g., `_chr`).
- If you prefer to store checkpoints under the repo (e.g., `FakeImageDetection/checkpoints/`), adjust the save/load paths in `train.py` and `test.py` accordingly.

## &#9733; Testing Script (test.py)

### Description
`test.py` evaluates trained models on the Wang_CVPR2020 and Ojha_CVPR2023 datasets. Metrics reported per sub-dataset are Average Precision (AP), Accuracy (Acc), and AUC (all as percentages). When `--data_type both` is used, results are saved into a single file under `results/both`, and an `Overall Averages` row is appended at the end.

### Basic Command

```
python -m torch.distributed.launch --nproc_per_node=GPU_NUM test.py -- [options]
```

Key options:
```
--model_name    : RN50, RN50_mod, RN50_npr, CLIP_vitl14, MNv2, SWIN_t, VGG11
--mask_type     : fourier, cosine, wavelet, pixel, patch, translate, rotate, rotate_translate, nomask
--band          : all, low, mid, high, low+mid, low+high, mid+high (optionally append +prog for progressive masking)
--mask_channel  : all, r, g, b, 0, 1, 2 (applies to frequency masks only)
--combine_aug   : none, rotate, translate, rotate_translate (applies in addition to frequency masks)
--ratio         : integer percent (e.g., 15)
--batch_size    : default 64
--data_type     : Wang_CVPR20, Ojha_CVPR23, or both (default both)
```

### Bash Command
Example testing launcher `test.sh`:

```
#!/bin/bash

# Define the arguments for your test script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
DATA_TYPE="both"  # both, or Wang_CVPR20, or Ojha_CVPR23
MODEL_NAME="RN50" # RN50, RN50_mod, RN50_npr, CLIP_vitl14, MNv2, SWIN_t, VGG11
MASK_TYPE="fourier"   # nomask, fourier, pixel, patch, cosine, wavelet, translate, rotate, rotate_translate
BAND="all" # all, low, mid, high, low+mid, low+high, mid+high
RATIO=15 # automatically becomes RATIO=0 if MASK_TYPE="nomask"
BATCH_SIZE=64
MASK_CHANNEL="all"    # all, r, g, b, 0, 1, 2 (applies to fourier/cosine/wavelet)
COMBINE_AUG="rotate"   # none, rotate, translate, rotate_translate (combine with frequency masking)

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs
echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Randomize master port between 29000 and 29999 to avoid clashes
MASTER_PORT=$((29000 + RANDOM % 1000))
echo "Using master port: $MASTER_PORT"

# Run the test command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$MASTER_PORT test.py \
  -- \
  --data_type $DATA_TYPE \
  --pretrained \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --mask_channel $MASK_CHANNEL \
  --combine_aug $COMBINE_AUG \
  --batch_size $BATCH_SIZE \
```

Now, use this to run testing:
```
bash test.sh "0" # gpu id/s to use
```

### Results
Results are saved under:

```
results/
└── both/
    └── rn50ft_fouriermask_rotate_chr15.txt   # example filename
```

- Each file includes the header, the per-dataset rows formatted as `Dataset, Avg.Prec.(%), Acc.(%), AUC(%)`, and an `AVERAGE` line summarizing all evaluated datasets in the run.


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
echo "The current date is: $current_date"

# Define the arguments for your training script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
NUM_EPOCHS=10000
PROJECT_NAME="Frequency-Masking"
MODEL_NAME="RN50" # RN50, RN50_mod, RN50_npr, CLIP_vitl14, MNv2, SWIN_t, VGG11
MASK_TYPE="rotate_translate" # nomask, fourier, pixel, patch, cosine, wavelet, translate, rotate, rotate_translate
BAND="all" # all, low, mid, high, low+mid, low+high, mid+high  ##### add +prog if using progressive masking
RATIO=15
BATCH_SIZE=128
MASK_CHANNEL="all" # all, r, g, b, 0, 1, 2 (applies to fourier/cosine/wavelet)
COMBINE_AUG="none" # none, rotate, translate, rotate_translate (combine with frequency masking)
WANDB_ID="2w0btkas"
RESUME="from_last" # from_last or from_best
learning_rate=0.0001 # 0.0001 * NUM_GPU

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs
echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Randomize master port between 29000 and 29999 to avoid clashes
MASTER_PORT=$((29000 + RANDOM % 1000))
echo "Using master port: $MASTER_PORT"

# Run the distributed training command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$MASTER_PORT train.py \
  -- \
  --num_epochs $NUM_EPOCHS \
  --project_name $PROJECT_NAME \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --mask_channel $MASK_CHANNEL \
  --combine_aug $COMBINE_AUG \
  --lr ${learning_rate} \
  --batch_size $BATCH_SIZE \
  --early_stop \
  --pretrained \
  # --resume_train $RESUME \
  # --clip_grad \
  # --debug \
```

Now, use this to run training:
```bash
bash train.sh "0,1,2,4" # gpu ids to use
```

Important:
- When starting the training (from 1st epoch), you can comment out `--resume_train $RESUME` and `--clip_grad`/`--debug`. If you don't want Weights & Biases logging, keep `--wandb_online` commented out and avoid setting `--wandb_run_id`.
- If training stalls mid-epoch, interrupt with Ctrl+C and resume from the last or best epoch using `--resume_train`.

## ★ Notes & Known Issues

- Paths for datasets and checkpoints are hardcoded under `/mnt/SCRATCH/chadolor/...` in multiple scripts (`train.py`, `test.py`). Please update for your environment.
- CLIP freezing bug in `networks/clip_models.py`:
  - In `CLIPModel.__init__()`, `param.requires_grad = freeze` should use the `unfreeze` argument (e.g., `param.requires_grad = unfreeze`).
- CLIP checkpoint saving/loading consistency:
  - `earlystop.py` saves only `fc` when `model_name == 'clip'`, while other parts use names like `CLIP_vitl14`. Align naming or condition (e.g., check `'CLIP' in model_name`).
- `prune.py` is provided as a pruning utility and may require adjustments (paths, arguments) to match your environment.
- Distributed launch: scripts currently use `torch.distributed.launch`. Consider `torchrun` for newer PyTorch versions.

## &#9733; License

This project is licensed under the [Apache License](LICENSE).

## &#9733; Citation

If you use this code in your research, please consider citing it. Below is the BibTeX entry for citation:

```bibtex
@article{Doloriel2024FrequencyMF,
  title={Frequency Masking for Universal Deepfake Detection},
  author={Chandler Timm C. Doloriel and Ngai-Man Cheung},
  journal={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  pages={13466-13470},
  url={https://api.semanticscholar.org/CorpusID:266977102}
}



## Training Script (train.py)

### Description

This script `(train.py)` is designed for distributed training and evaluation of various Deep Learning models including ResNet variants. The script is highly configurable through command-line arguments and provides advanced features such as `WandB` integration, early stopping, and various masking options for data augmentation.

### Basic Command

To run the script in a distributed environment:

```bash
python -m torch.distributed.launch --nproc_per_node=GPU_NUM train.py -- [options]

```

### Command-Line Options

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

## Testing Script (test.py)

### Description
The script `test.py` is designed for evaluating trained models on multiple datasets. The script leverages metrics such as Average Precision, Accuracy, and Area Under the Curve (AUC) for evaluation.

### Usage
Basic Command

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

## FrequencyMaskGenerator Class

### Description
The `FrequencyMaskGenerator` class is designed for applying frequency domain masking to images. This is particularly useful as an advanced data augmentation technique, where the frequency components of an image are selectively blocked or passed.

### Key Features

- Frequency Domain Transformation: Converts the input image to its frequency representation using the Fast Fourier Transform (FFT).
- Mask Generation: Creates a balanced binary mask in the frequency domain. The ratio of frequencies to be masked can be specified.
- Inverse Transformation: Converts the masked frequency representation back to the spatial domain.

### Methods
- `__init__(self, ratio: float = 0.3):` Initializes the mask generator. The ratio parameter specifies the fraction of frequencies to be masked.

- `transform(self, image: Image.Image) -> Image.Image:` Transforms an input PIL Image by applying a frequency mask and returns the masked image.

- `_create_balanced_mask(self, height, width):` Internal method to create a balanced binary mask based on the image dimensions (height and width).

### Usage Example
```python
from PIL import Image
from FrequencyMaskGenerator import FrequencyMaskGenerator

# Initialize the FrequencyMaskGenerator with a ratio of 0.5
mask_generator = FrequencyMaskGenerator(ratio=0.5)

# Read an image using PIL
image = Image.open("sample_image.jpg")

# Apply the frequency mask
masked_image = mask_generator.transform(image)

# Save or display the masked image
masked_image.show()

```
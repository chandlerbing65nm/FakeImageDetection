


## Training Script (train.py)

### Description

This script `(train.py)` is designed for distributed training and evaluation of various Deep Learning models including ResNet and Vision Transformer (ViT) variants. The script is highly configurable through command-line arguments and provides advanced features such as `WandB` integration, early stopping, and various masking options for data augmentation.

### Key Features

- Distributed Training using `torch.distributed`
- Support for multiple model architectures including ResNet and ViT
- Data Augmentation using custom mask generators
- Loss function: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- Optimizer: AdamW with learning rate of 0.0001 and weight decay of 1e-4

### Basic Command

To run the script in a distributed environment:

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py -- [options]

```

### Command-Line Options

```bash
--local_rank     : Local rank for distributed training. Default is 0.
--num_epochs     : Number of epochs for training. Default is 2.
--model_name     : Type of the model. Choices include various ResNet and ViT variants.
--wandb_online   : Run WandB in online mode. Default is offline.
--project_name   : Name of the WandB project. Default is "Masked-ResNet".
--wandb_run_id   : WandB run ID.
--resume_train   : Resume training from last or best epoch.
--pretrained     : Use pretrained model.
--early_stop     : Enable early stopping.
--mask_type      : Type of mask generator for data augmentation. Choices include 'zoom', 'patch', 'spectral', etc.
--batch_size     : Batch size for training. Default is 64.
--ratio          : Masking ratio for data augmentation. Default is 50.
```

### Examples
Run with ResNet-50 and spectral mask:

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py -- --model_name=RN50 --mask_type=spectral
```
or
```bash
bash train.sh
```

## Testing Script (test.py)

### Description
The script `test.py` is designed for evaluating trained models on multiple datasets. It supports both ResNet and Vision Transformer (ViT) architectures and various masking strategies. The script leverages advanced metrics such as Average Precision, Accuracy, and Area Under the Curve (AUC) for evaluation.

### Key Features
- Support for multiple model architectures including ResNet and ViT
- Evaluation on multiple datasets including `Wang_CVPR20` and `Ojha_CVPR23`
- Metrics: Average Precision, Accuracy, and AUC
- Data Augmentation using custom mask generators

### Usage
Basic Command

```bash
python test.py [options]
```
Command-Line Options
```bash
--model_name : Type of the model. Choices include various ResNet and ViT variants.
--mask_type  : Type of mask generator for data augmentation. Choices include 'zoom', 'patch', 'spectral', etc.
--pretrained : Use pretrained model.
--ratio      : Masking ratio for data augmentation.
--batch_size : Batch size for evaluation. Default is 64.
--data_type  : Type of dataset for evaluation. Choices are 'Wang_CVPR20' and 'Ojha_CVPR23'.
--device     : Device to use for evaluation (default: auto-detect).
```

### Examples
Evaluate a ResNet-50 model with spectral masking on the Wang_CVPR20 dataset:
```bash
python test.py --model_name=RN50 --mask_type=spectral --data_type=Wang_CVPR20
```
or
```bash
bash test.sh
```

## FrequencyMaskGenerator Class

## Description
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
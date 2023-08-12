import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import numpy as np
from PIL import Image
import os
import clip
from tqdm import tqdm
import argparse

from dataset import WangEtAlDataset, CorviEtAlDataset, ForenSynths, GenImage
from extract_features import *
from classifier_train import MLPClassifier, SelfAttentionClassifier, LinearClassifier

def evaluate_model(data_type, clip_model, dataset_path, model_choice, classifier_params, checkpoint_path, device):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    if data_type == 'GenImage':
        test_dataset = GenImage(dataset_path, transform=transform)
    elif data_type == 'ForenSynths' :
        test_dataset = ForenSynths(dataset_path, transform=transform)
    else:
        raise ValueError("wrong dataset input")

    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    feature_extractor = CLIPFeatureExtractor(model_name=clip_model, device=device)
    # Extract the features
    test_real_embeddings, test_fake_embeddings = feature_extractor.extract_features(test_dataloader)

    feature_size = test_fake_embeddings.shape[1]

    if model_choice == "linear":
        model = LinearClassifier(input_size=feature_size).to(device)
    elif model_choice == "mlp":
        model = MLPClassifier(input_size=feature_size).to(device)
    elif model_choice == "attention":
        model = SelfAttentionClassifier(input_size=feature_size, **classifier_params).to(device)

    model.load_state_dict(torch.load(checkpoint_path))

    # Handle the possibility that there might be only one class in the dataset
    if test_real_embeddings.size != 0 and test_fake_embeddings.size != 0:
        # Both 'real' and 'fake' classes are present
        test_embeddings = np.concatenate((test_real_embeddings, test_fake_embeddings), axis=0)
        test_labels = np.array([0] * len(test_real_embeddings) + [1] * len(test_fake_embeddings))
    else:
        # Only 'fake' class is present
        test_embeddings = test_fake_embeddings
        test_labels = np.array([1] * len(test_fake_embeddings))
    
    model.cuda()    
    model.eval()  # set model to evaluation mode

    # Lists to store metrics for each run
    accuracies, average_precisions = [], []

    for _ in range(3):

        test_data = TensorDataset(torch.tensor(test_embeddings, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32))
        test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=4)

        with torch.no_grad():
            y_true, y_pred = [], []
            for img, label in tqdm(test_loader, "accessing test dataloader"):
                in_tens = img.cuda()
                y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

        y_true, y_pred = np.array(y_true), np.array(y_pred)

        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)

        accuracies.append(acc)
        average_precisions.append(ap)

    avg_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    avg_ap = np.mean(average_precisions)
    std_ap = np.std(average_precisions)

    # Print results
    print(f'Average Precision: {avg_ap}, Std Dev: {std_ap}')
    print(f'Accuracy: {avg_acc}, Std Dev: {std_acc}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for your script")

    parser.add_argument(
        '--device', 
        default="cuda:0" if torch.cuda.is_available() else "cpu", 
        type=str, 
        help="Device to use (default: auto-detect)"
        )
    parser.add_argument(
        '--data_type', 
        default="ForenSynths", 
        type=str, 
        choices=['GenImage', 'ForenSynths'], 
        help="Dataset Type"
        )
    parser.add_argument(
        '--clip_model', 
        default='ViT-L/14', 
        type=str, 
        choices=['ViT-B/16', 'ViT-L/14', 'RN50', 'RN101'],
        help='Type of clip visual model'
        )
    parser.add_argument(
        '--model_choice', 
        default="linear", 
        type=str, 
        choices=['attention', 'mlp', 'linear'], 
        help="Model choice"
        )
    parser.add_argument('--nhead', default=8, type=int, help="Number of heads for attention model")
    parser.add_argument('--num_layers', default=6, type=int, help="Number of layers for attention model")
    parser.add_argument(
        '--mask_ratio', 
        type=int, 
        default=50,
        help='Ratio of mask to apply'
        )
    parser.add_argument(
        '--mask_type', 
        default='zoom', 
        choices=['zoom', 'patch', 'spectral', 'shiftedpatch', 'nomask'], 
        help='Type of mask generator'
        )

    args = parser.parse_args()
    clip_model = args.clip_model.lower().replace('/', '').replace('-', '')

    device = torch.device(args.device)
    data_type = args.data_type
    model_choice = args.model_choice
    classifier_params = {"nhead": args.nhead, "num_layers": args.num_layers}

    if args.mask_type in ['zoom', 'patch', 'spectral', 'shiftedpatch'] and args.mask_ratio > 0:
        mask_ratio = args.mask_ratio
        checkpoint_path = f'checkpoints/mask_{mask_ratio}/{clip_model}_{args.mask_type}maskclip_best_{model_choice}.pth'
    else:
        mask_ratio = 0
        checkpoint_path = f'checkpoints/mask_{mask_ratio}/{clip_model}_clip_best_{model_choice}.pth'

    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Device: {args.device}")
    print(f"Dataset Type: {args.data_type}")
    print(f"CLIP model type: {args.clip_model}")
    print(f"Model Choice: {args.model_choice}")
    print(f"Ratio of mask: {mask_ratio}")
    print(f"Mask Type: {args.mask_type}")
    print(f"Number of Heads: {args.nhead}")
    print(f"Number of Layers: {args.num_layers}")
    print(f"Checkpoint Type: {checkpoint_path}")
    print("-" * 30, "\n")

    if args.data_type == 'ForenSynths':
        dataset_paths = [
            '../../Datasets/Wang_CVPR20/progan',
            '../../Datasets/Wang_CVPR20/cyclegan',
            '../../Datasets/Wang_CVPR20/biggan',
            '../../Datasets/Wang_CVPR20/stylegan',
            '../../Datasets/Wang_CVPR20/gaugan',
             '../../Datasets/Wang_CVPR20/stargan',
            '../../Datasets/Wang_CVPR20/deepfake',
            '../../Datasets/Wang_CVPR20/seeingdark',
            '../../Datasets/Wang_CVPR20/san',
            '../../Datasets/Wang_CVPR20/crn',
            '../../Datasets/Wang_CVPR20/imle',
            ]
    elif args.data_type == 'GenImage':
        dataset_paths = [
            '../../Datasets/GenImage/imagenet_vqdm/imagenet_vqdm/val',
            '../../Datasets/GenImage/imagenet_glide/imagenet_glide/val',
            ]
    else:
        raise ValueError("wrong dataset type")

    for dataset_path in dataset_paths:
        print(f"\nEvaluating dataset at path: {dataset_path}")

        evaluate_model(
            data_type=data_type,
            clip_model=args.clip_model,
            dataset_path=dataset_path,
            model_choice=model_choice,
            classifier_params=classifier_params,
            checkpoint_path=checkpoint_path,
            device=device,
        )
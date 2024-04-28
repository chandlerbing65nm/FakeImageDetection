
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
import torch.distributed as dist  # If distributed training is being used
import wandb  # If Weights & Biases is being used for logging

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
import timm 
import copy
import pprint
from torchprofile import profile_macs

from dataset import *
from augment import ImageAugmentor
from mask import *
from utils import *
from networks.resnet import resnet50
from networks.resnet_mod import resnet50 as _resnet50, ChannelLinear

from networks.clip_models import CLIPModel
import time

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import torch.nn.utils.prune as prune
import shap

from commons import get_model_flops, get_model_sparsity

os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'


def calculate_model_sparsity(model):
    total_zeros = 0
    total_elements = 0
    for module in model.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            weight = module.weight.data
            total_zeros += torch.sum(weight == 0).item()
            total_elements += weight.nelement()
        if hasattr(module, 'bias') and module.bias is not None:
            bias = module.bias.data
            total_zeros += torch.sum(bias == 0).item()
            total_elements += bias.nelement()

    overall_sparsity = total_zeros / total_elements
    print(f"model sparsity: {overall_sparsity:.4f} ({total_zeros}/{total_elements})")
    return overall_sparsity

def read_prune_amounts(filename):
    with open(filename, 'r') as file:
        for line in file:
            # Skip commented lines
            if line.strip().startswith('#'):
                continue
            # Process the first non-commented line
            amounts = line.strip('[] \n').split(',')
            return [float(amount.strip()) for amount in amounts]
    raise ValueError("No valid pruning amounts found in the file.")


def iterative_pruning_finetuning(
    model, 
    criterion, 
    optimizer, 
    scheduler,
    train_loader, 
    val_loader, 
    device,
    learning_rate,
    num_epochs_per_pruning = 5,
    save_path='./',
    args=None,
    ):
    
    '''
    args.pruning_rounds - number of pruning rounds
    num_epochs_per_pruning - number of epochs per pruning round
    '''

    for i in range(args.pruning_rounds):
        if dist.get_rank() == 0:
            print("Pruning...")
        
        dist.barrier() # Synchronize all processes


        if args.pruning_test == True: # for pruning the eval model
            original_model = copy.deepcopy(model)  # Keep a copy of the original model to reset after each pruning

            # Gather all Conv2d layers
            conv2d_layers = [(name, module) for name, module in original_model.named_modules() 
                    if isinstance(module, torch.nn.Conv2d) and "downsample" not in name]

            for layer_name, layer_module in conv2d_layers:
                # Reset model to original state for each layer
                model = copy.deepcopy(original_model)

                # Access the same layer in the current model copy
                current_layer = dict(model.named_modules())[layer_name]

                # Before pruning
                if dist.get_rank() == 0:
                    print("Before pruning:")
                    sparsity = calculate_model_sparsity(model)
                    if sparsity > 1e-3:  # This allows a tiny tolerance
                        raise ValueError(f"Error: Expected sparsity to be 0 before pruning, but got {sparsity:.12f}")


                # Apply pruning only to the current Conv2d layer
                prune.ln_structured(
                    current_layer, name="weight",
                    amount=amount, n=2, dim=0
                )

                # After pruning
                if dist.get_rank() == 0:
                    print(f"After pruning {layer_name}:")
                    calculate_model_sparsity(model)

                # Evaluate the pruned model
                model.eval()
                data_loader = val_loader

                total_samples = len(data_loader.dataset)
                running_loss = 0.0
                y_true, y_pred = [], []

                for batch_data in tqdm(data_loader, f"Test after Pruning layer {layer_name}", disable=dist.get_rank() != 0):
                    batch_inputs, batch_labels = batch_data
                    batch_inputs = batch_inputs.to(device)
                    batch_labels = batch_labels.float().to(device)

                    if 'clip' in args.model_name and args.clip_grad == True:
                        outputs = model(batch_inputs, return_all=True).view(-1).unsqueeze(1)
                    else:
                        outputs = model(batch_inputs).view(-1).unsqueeze(1) # pass the input to the fc layer only

                    y_pred.extend(outputs.sigmoid().detach().cpu().numpy())
                    y_true.extend(batch_labels.cpu().numpy())

                y_true, y_pred = np.array(y_true), np.array(y_pred)
                acc = accuracy_score(y_true, y_pred > 0.5)
                ap = average_precision_score(y_true, y_pred)

                if dist.get_rank() == 0:
                    print(f'Layer {layer_name} -> Acc: {acc:.4f} AP: {ap:.4f}')

                    ##################### Evaluate Model on Each Dataloader ####################
                    with open("pruning_results.txt", "a") as file:  # Open file in append mode
                        if layer_name == 'module.conv1' or layer_name == 'module.model.visual.conv1':
                            # Write a header at the start of the file if this is the first index
                            file.write("\n\n" + "-" * 28 + "\n")
                            file.write(f"mAP of {args.band} band {args.mask_type} masking ({args.ratio}%) of {args.model_name} pruned at {args.conv2d_prune_amount * 100:.1f}% --> ImageNet weights?: {args.pretrained}\n")
                            if args.pretrained == False: file.write(f"Weights loaded from: {args.checkpoint_path}\n")
                            file.write(f"Dataset: {args.dataset}\n")
                        file.write(f"{ap:.4f}" + "\n")  # Write each result immediately to the file

                    print("##########################\n")
            
        elif args.pruning_ft == True: # for pruning then finetuning
            # Gather all Conv2d layers
            conv2d_layers = [(name, module) for name, module in model.named_modules() 
                    if isinstance(module, torch.nn.Conv2d) and "downsample" not in name]

            # Read prune amounts from file
            prune_amounts = read_prune_amounts(args.conv2d_prune_amount_file)
            assert len(conv2d_layers) == len(prune_amounts), "The number of layers and prune amounts must match"

            for (layer_name, layer_module), amount in zip(conv2d_layers, prune_amounts):

                # Access the same layer in the current model copy
                current_layer = dict(model.named_modules())[layer_name]

                # Before pruning
                if dist.get_rank() == 0:
                    if layer_name == 'module.conv1' or layer_name == 'module.model.visual.conv1':
                        print(f"Before pruning {layer_name}:")
                        sparsity = calculate_model_sparsity(model)
                        if sparsity > 1e-3:  # This allows a tiny tolerance
                            raise ValueError(f"Error: Expected sparsity to be 0 before pruning, but got {sparsity:.12f}")

                # Apply pruning only to the current Conv2d layer
                prune.ln_structured(
                    current_layer, name="weight",
                    amount=amount, n=2, dim=0
                )

                # After pruning
                if dist.get_rank() == 0:
                    print(f"After pruning {layer_name}:")
                    calculate_model_sparsity(model)

            model = train_model(
                model, criterion, optimizer, 
                scheduler, train_loader, val_loader, 
                num_epochs=num_epochs_per_pruning, 
                resume_epoch=0, save_path=save_path,
                early_stopping=None, device=device,
                args=args,
                )
            model = copy.deepcopy(model)

        dist.barrier() # Synchronize all processes

        model = remove_parameters(model) # remove the pruner masks and hooks - this does not undo the magnitude pruning

        # Save the model after pruning/fine-tuning
        if dist.get_rank() == 0 and args.pruning_test != True:
            state = {
                'model_state_dict': model.state_dict(),
            }

            save_dir="./checkpoints/pruning"
            os.makedirs(save_dir, exist_ok=True)
            if args.pruning_ft == True:
                model_save_path = os.path.join(save_dir, f"lnstructured.pth")
                torch.save(state, model_save_path)
                print(f"\n\nPruned model saved to {model_save_path}")

    return model


def train_model(
    model, 
    criterion, 
    optimizer, 
    scheduler,
    train_loader, 
    val_loader, 
    num_epochs=25, 
    resume_epoch=0,
    save_path='./', 
    early_stopping=None,
    device='cpu',
    sampler=None,
    args=None,
    ):

    features_path = f"features.pth"
    features_exist = os.path.exists("./clip_train_" + features_path)

    for epoch in range(resume_epoch, num_epochs):
        if epoch % 1 == 0:  # Only print every 20 epochs
            if dist.get_rank() == 0:
                print('\n')
                print(f'Epoch {epoch}/{num_epochs}')
                print('-' * 10)

        # For CLIP model, extract features only once
        if 'clip' in args.model_name and not features_exist:
            # Process with rank 0 performs the extraction
            if dist.get_rank() == 0:
                extract_and_save_features(model, train_loader, "./clip_train_" + features_path, device)
                extract_and_save_features(model, val_loader, "./clip_val_" + features_path, device)
                # Create a temporary file to signal completion
                with open(f'clip_extract.done', 'w') as f:
                    f.write('done')
            
            # Other processes wait until the .done file is created
            else:
                while not os.path.exists(f'clip_extract.done'):
                    time.sleep(5)  # Sleep to avoid busy waiting

            features_exist = True  # Set this to True after extraction

        # Load the features for all processes if not done already
        if 'clip' in args.model_name and features_exist and epoch == resume_epoch:
            train_loader = load_features("./clip_train_" + features_path, batch_size=args.batch_size, shuffle=False)
            val_loader = load_features("./clip_val_" + features_path, batch_size=args.batch_size, shuffle=False)

            # Assuming files can be safely deleted after loading
            import os
            os.remove("./clip_train_" + features_path)
            os.remove("./clip_val_" + features_path)
            os.remove("clip_extract.done")

        phases = ['Validation', 'Training', 'Validation'] if epoch==resume_epoch else ['Training', 'Validation']
        # Iterate through the phases
        for phase in phases:
            if phase == 'Training':
                if sampler is not None:
                    sampler.set_epoch(epoch)
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            total_samples = len(data_loader.dataset)
            running_loss = 0.0
            y_true, y_pred = [], []

            # disable_tqdm = dist.get_rank() != 0
            # data_loader_with_tqdm = tqdm(data_loader, f"{phase}", disable=disable_tqdm)

            for batch_data in tqdm(data_loader, f"{phase}", disable=dist.get_rank() != 0):
                batch_inputs, batch_labels = batch_data
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.float().to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(batch_inputs).view(-1).unsqueeze(1)
                    loss = criterion(outputs.squeeze(1), batch_labels)

                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * batch_inputs.size(0)
                y_pred.extend(outputs.sigmoid().detach().cpu().numpy())
                y_true.extend(batch_labels.cpu().numpy())

            epoch_loss = running_loss / total_samples

            y_true, y_pred = np.array(y_true), np.array(y_pred)
            acc = accuracy_score(y_true, y_pred > 0.5)
            ap = average_precision_score(y_true, y_pred)

            if epoch % 1 == 0:  # Only print every epoch
                if dist.get_rank() == 0:
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {acc:.4f} AP: {ap:.4f}')

            # Early stopping
            if phase == 'Validation':
                # if dist.get_rank() == 0:
                #     wandb.log({"Validation Loss": epoch_loss, "Validation Acc": acc, "Validation AP": ap}, step=epoch)
                
                if args.pruning_ft==False:
                    early_stopping(acc, model, optimizer, epoch)  # Pass the accuracy instead of loss
                    if early_stopping.early_stop:
                        if dist.get_rank() == 0:
                            print("Early stopping")
                        return model
            else:
                if args.pruning_ft:
                    scheduler.step()

                # if dist.get_rank() == 0:
                #     wandb.log({"Training Loss": epoch_loss, "Training Acc": acc, "Training AP": ap}, step=epoch)

    return model


def evaluate_model(
    model_name,
    data_type,
    mask_type, 
    ratio,
    dataset_path, 
    batch_size,
    checkpoint_path, 
    device,
    args
    ):

    # Depending on the mask_type, create the appropriate mask generator
    if mask_type == 'spectral':
        mask_generator = FrequencyMaskGenerator(ratio=ratio)
    elif mask_type == 'pixel':
        mask_generator = PixelMaskGenerator(ratio=ratio)
    else:
        mask_generator = None


    test_opt = {
        'rz_interp': ['bilinear'],
        'loadSize': 256,
        'blur_prob': 0.1,  # Set your value
        'blur_sig': [(0.0 + 3.0) / 2],
        'jpg_prob': 0.1,  # Set your value
        'jpg_method': ['pil'],
        'jpg_qual': [int((30 + 100) / 2)]
    }

    test_transform = test_augment(ImageAugmentor(test_opt), mask_generator, args)

    if data_type == 'GenImage':
        test_dataset = GenImage(dataset_path, transform=test_transform)
    elif data_type == 'Wang_CVPR20' :
        test_dataset = Wang_CVPR20(dataset_path, transform=test_transform)
    elif data_type == 'Ojha_CVPR23' :
        test_dataset = OjhaCVPR23(dataset_path, transform=test_transform)
    else:
        raise ValueError("wrong dataset input")

    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=4)

    if model_name == 'RN50':
        # model = torchvision.models.resnet50(pretrained=pretrained)
        # model.fc = nn.Linear(model.fc.in_features, 1)
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'RN50_mod':
        model = _resnet50(pretrained=False, stride0=1)
        model.fc = ChannelLinear(model.fc.in_features, 1)
    elif model_name.startswith('ViT'):
        model_variant = model_name.split('_')[1] # Assuming the model name is like 'ViT_base_patch16_224'
        model = timm.create_model(model_variant, pretrained=pretrained)
    elif model_name == 'clip_vitl14':
        clip_model_name = 'ViT-L/14'
        model = CLIPModel(clip_model_name, num_classes=1)
    elif model_name == 'clip_rn50':
        clip_model_name = 'RN50'
        model = CLIPModel(clip_model_name, num_classes=1)
    else:
        raise ValueError(f"Model {model_name} not recognized!")

    model = model.to(device)
    model = DistributedDataParallel(model, find_unused_parameters=True)

    checkpoint = torch.load(checkpoint_path)

    if 'clip' in args.model_name and args.other_model != True and args.clip_ft == False:
        model.module.fc.load_state_dict(checkpoint['model_state_dict'])
    elif args.other_model:
        model.module.fc.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval() 

    # Calculate FLOPs
    if dist.get_rank() == 0 and dataset_path.endswith(('progan', 'guided')):
        print(f"\n\nModel FLOPs Ratio: {get_model_flops(model) * 100:.2f}")
        print(f'Sparsity: {get_model_sparsity(model) * 100:.2f}\n\n')

    y_true, y_pred = [], []

    disable_tqdm = dist.get_rank() != 0
    data_loader_with_tqdm = tqdm(test_dataloader, "test dataloading", disable=disable_tqdm)

    with torch.no_grad():
        for inputs, labels in data_loader_with_tqdm:
            inputs = inputs.to(device)

            labels = labels.float().to(device)
            if 'clip' in args.model_name:
                outputs = model(inputs, return_all=True).view(-1).unsqueeze(1)
            else:
                outputs = model(inputs).view(-1).unsqueeze(1)
            y_pred.extend(outputs.sigmoid().detach().cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    if dist.get_rank() == 0:
        print(f'Average Precision: {ap}')
        print(f'Accuracy: {acc}')
        print(f'ROC AUC Score: {auc}')

    return ap, acc, auc

def train_augment(augmentor, mask_generator=None, args=None):
    transform_list = []
    if mask_generator is not None:
        transform_list.append(transforms.Lambda(lambda img: mask_generator.transform(img)))
    transform_list.extend([
        transforms.Lambda(lambda img: augmentor.custom_resize(img)),
        transforms.Lambda(lambda img: augmentor.data_augment(img)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if args is not None and 'clip' in args.model_name:
        transform_list.append(transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
    else:
        transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(transform_list)

def val_augment(augmentor, mask_generator=None, args=None):
    transform_list = []
    if mask_generator is not None:
        transform_list.append(transforms.Lambda(lambda img: mask_generator.transform(img)))
    transform_list.extend([
        transforms.Lambda(lambda img: augmentor.custom_resize(img)),
        transforms.Lambda(lambda img: augmentor.data_augment(img)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    if args is not None and 'clip' in args.model_name:
        transform_list.append(transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
    else:
        transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(transform_list)

def test_augment(augmentor, mask_generator=None, args=None):
    transform_list = [
        # transforms.Lambda(lambda img: augmentor.custom_resize(img)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    if args is not None and 'clip' in args.model_name:
        transform_list.append(transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
    else:
        transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(transform_list)


def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model

def extract_and_save_features(model, data_loader, save_path, device='cpu'):
    model.eval()
    features = []
    labels_list = []

    # disable_tqdm = dist.get_rank() != 0
    # data_loader_with_tqdm = tqdm(data_loader, "Extracting CLIP Features", disable=dist.get_rank() != 0)

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, "Extracting CLIP Features", disable=dist.get_rank() != 0):
            inputs = inputs.to(device)
            features.append(model(inputs, return_feature=True).detach().cpu())
            labels_list.append(labels)

    features = torch.cat(features)
    labels = torch.cat(labels_list)
    torch.save((features, labels), save_path)

def load_features(save_path, batch_size=32, shuffle=True):
    features, labels = torch.load(save_path)
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

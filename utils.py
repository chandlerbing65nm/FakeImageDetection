
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

import torchvision.models as vis_models

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


def iterative_pruning_finetuning(
    model, 
    criterion, 
    optimizer, 
    scheduler,
    train_loader, 
    val_loader, 
    device,
    learning_rate,
    num_epochs_per_pruning = 1,
    save_path='./',
    args=None,
    ):
    
    '''
    args.pruning_rounds - number of pruning rounds
    num_epochs_per_pruning - number of epochs per pruning round
    '''

    for i in range(args.pruning_rounds):
        if dist.get_rank() == 0:
            # print("\nPruning and Finetuning {}/{}".format(i + 1, args.pruning_rounds))
            print("Pruning...")
        
        dist.barrier() # Synchronize all processes

        # NOTE: For global pruning, linear/dense layer can also be pruned!
        if args.global_prune == True:
            # args.global_prune -> Global pruning
            parameters_to_prune = []
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))
                elif isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, "weight"))
        
            # L1Unstructured - prune (currently unpruned) entries in a tensor by zeroing
            # out the ones with the lowest absolute magnitude-
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method = prune.L1Unstructured, # L1Unstructured or LnStructured
                amount = args.conv2d_prune_amount,
            )
            # Compute global sparsity
            _, _, sparsity = measure_global_sparsity(
                model, 
                weight = True,
                bias = True, 
                conv2d_use_mask = True,
                linear_use_mask = False
                )
            if dist.get_rank() == 0:
                print(f"\nPruning Iter: {i+1}, Global Sparsity: {sparsity * 100:.3f}%")
        # layer-wise pruning-
        else:
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
                    if sparsity != 0:
                        raise ValueError(f"Error: Expected sparsity to be 0 before pruning, but got {sparsity:.4f}")

                # Apply pruning only to the current Conv2d layer
                prune.ln_structured(
                    current_layer, 
                    name="weight",
                    amount=args.conv2d_prune_amount,
                    n=2,
                    dim=0
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

                    outputs = model(batch_inputs).view(-1).unsqueeze(1)
                    y_pred.extend(outputs.sigmoid().detach().cpu().numpy())
                    y_true.extend(batch_labels.cpu().numpy())

                y_true, y_pred = np.array(y_true), np.array(y_pred)
                acc = accuracy_score(y_true, y_pred > 0.5)
                ap = average_precision_score(y_true, y_pred)

                if dist.get_rank() == 0:
                    print(f'Layer {layer_name} -> Acc: {acc:.4f} AP: {ap:.4f}')

                ##################### Evaluate Model on Each Dataloader ####################
                with open("pruning_results.txt", "a") as file:  # Open file in append mode
                    if layer_name == 'module.conv1':
                        # Write a header at the start of the file if this is the first index
                        file.write("\n\n" + "-" * 28 + "\n")
                        file.write(f"mAP of {args.band} band {args.mask_type} masking ({args.ratio}%) of {args.model_name} pruned at {args.conv2d_prune_amount * 100:.1f}% --> ImageNet weights?: {args.pretrained}\n")
                        if args.pretrained == False:
                            file.write(f"Weights loaded from: {args.checkpoint_path}\n")
                    file.write(f"{ap:.4f}" + "\n")  # Write each result immediately to the file

                print("##########################\n")

        dist.barrier() # Synchronize all processes

        if args.pruning_ft == True:
            model = train_model(
                model, 
                criterion, 
                optimizer, 
                scheduler,
                train_loader, 
                val_loader, 
                num_epochs=num_epochs_per_pruning, 
                resume_epoch=0,
                save_path=save_path,
                early_stopping=None,
                device=device,
                args=args,
                )
            model = copy.deepcopy(model)

        model = remove_parameters(model)

        # Save the model after pruning/fine-tuning
        if dist.get_rank() == 0 and args.pruning_test != True:
            state = {
                'model_state_dict': model.state_dict(),
            }

            if args.global_prune == True:
                save_dir="./checkpoints/pruned_globall1unstructured"
            else:
                save_dir="./checkpoints/pruned_layerwiselnstructured"

            os.makedirs(save_dir, exist_ok=True)

            if args.pruning_ft == True:
                model_save_path = os.path.join(save_dir, f"pruned+ft_model_{i+1}_pruned:{args.args.conv2d_prune_amount}.pth")
            else:
                model_save_path = os.path.join(save_dir, f"pruned_model_{i+1}_pruned:{args.args.conv2d_prune_amount}.pth")
            
            torch.save(state, model_save_path)
            print(f"Pruned model saved to {model_save_path}")

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
        if args.model_name == 'clip' and not features_exist:
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
        if args.model_name == 'clip' and features_exist and epoch == resume_epoch:
            train_loader = load_features("./clip_train_" + features_path, batch_size=args.batch_size, shuffle=False)
            val_loader = load_features("./clip_val_" + features_path, batch_size=args.batch_size, shuffle=False)

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
                        if args.pruning_ft:
                            l1_reg = torch.tensor(0.).to(device)
                            for module in model.modules():
                                mask = None
                                weight = None
                                for name, buffer in module.named_buffers():
                                    if name == "weight_mask":
                                        mask = buffer
                                for name, param in module.named_parameters():
                                    if name == "weight_orig":
                                        weight = param
                                # We usually only want to introduce sparsity to weights and prune weights.
                                # Do the same for bias if necessary.
                                if mask is not None and weight is not None:
                                    l1_reg += torch.norm(mask * weight, 1)

                            loss += 0 * l1_reg

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
                if dist.get_rank() == 0:
                    wandb.log({"Validation Loss": epoch_loss, "Validation Acc": acc, "Validation AP": ap}, step=epoch)
                
                if args.pruning_ft==False:
                    early_stopping(acc, model, optimizer, epoch)  # Pass the accuracy instead of loss
                    if early_stopping.early_stop:
                        if dist.get_rank() == 0:
                            print("Early stopping")
                        return model
            else:
                if args.pruning_ft:
                    scheduler.step()

                if dist.get_rank() == 0:
                    wandb.log({"Training Loss": epoch_loss, "Training Acc": acc, "Training AP": ap}, step=epoch)

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
    args,
    per_class_metrics=False
    ):

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

    test_transform = train_augment(ImageAugmentor(test_opt), mask_generator, args)


    if data_type == 'Wang_CVPR20' :
        test_dataset = Wang_CVPR20(dataset_path, transform=test_transform)
    elif data_type == 'Ojha_CVPR23' :
        test_dataset = OjhaCVPR23(dataset_path, transform=test_transform)
    elif data_type in ['ImageNet_mini']:
        # test_transform = transforms.Compose([ 
        #     transforms.Resize((224, 224)), 
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        test_dataset = torchvision.datasets.ImageFolder(dataset_path, transform=test_transform)
    else:
        raise ValueError("wrong dataset input")

    if args.conv_features or data_type == 'ImageNet':
        subset_size = int(0.1 * len(test_dataset))
        subset_indices = random.sample(range(len(test_dataset)), subset_size)
        test_dataset = Subset(test_dataset, subset_indices)

    test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=args.seed)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4)

    if model_name == 'RN50' and data_type in ['Wang_CVPR20', 'Ojha_CVPR23']:
        # model = vis_models.resnet50(pretrained=pretrained)
        # model.fc = nn.Linear(model.fc.in_features, 1)
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif data_type == 'ImageNet_mini':
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1000)
    elif model_name == 'RN50_mod':
        model = _resnet50(pretrained=True, stride0=1)
        model.fc = ChannelLinear(model.fc.in_features, 1)
    elif model_name.startswith('ViT'):
        model_variant = model_name.split('_')[1] # Assuming the model name is like 'ViT_base_patch16_224'
        model = timm.create_model(model_variant, pretrained=True)
    elif model_name.startswith('clip'):
        clip_model_name = 'ViT-L/14'
        model = CLIPModel(clip_model_name, num_classes=1)
        raise ValueError(f"Model {model_name} not recognized!")

    model = model.to(device)
    model = DistributedDataParallel(model)

    checkpoint = torch.load(checkpoint_path)

    if args.model_name=='clip' and args.other_model != True:
        model.module.fc.load_state_dict(checkpoint['model_state_dict'])
    elif args.other_model:
        model.module.fc.load_state_dict(checkpoint)
    # elif args.other_model:
    #     model.module.fc.load_state_dict(checkpoint)

    model = model.to(device)

    if args.conv_features and dist.get_rank() == 0:
        # Define a dictionary to hold features for each convolutional block, organized by stage and block
        features_dict = {stage: {block: [] for block in range(blocks)} for stage, blocks in enumerate([3, 4, 6, 3], start=1)}

        def hook_fn(module, input, output, stage, block):
            """Function to be called by hooks. It will save the output of each block."""
            features_dict[stage][block].append(output)

        def register_hooks(model):
            """Registers hooks on the convolutional blocks of the model."""
            # Check if the model is wrapped with DDP
            if hasattr(model, 'module'):
                # Access the underlying model
                model = model.module
            
            stages = [model.layer1, model.layer2, model.layer3, model.layer4]
            for stage_idx, stage in enumerate(stages, start=1):
                for block_idx, block in enumerate(stage.children(), start=0):
                    # The lambda function ensures that stage_idx and block_idx are captured correctly at hook registration time
                    block.register_forward_hook(lambda module, input, output, s=stage_idx, b=block_idx: hook_fn(module, input, output, s, b))

        # Register the hooks
        register_hooks(model)

    model.eval() 

    # Initialize containers for predictions and true labels
    y_true, y_pred = [], []

    # disable_tqdm = dist.get_rank() != 0
    # data_loader_with_tqdm = tqdm(test_dataloader, "test dataloading", disable=dist.get_rank() != 0):

    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, "test dataloading", disable=dist.get_rank() != 0):

            # inputs = torch.rand(inputs.shape[0], inputs.shape[1], 224, 224)  # Dummy inputs
            # labels = torch.randint(0, 1, (inputs.shape[0],))  # Dummy labels for classification

            inputs = inputs.to(device)
            labels = labels.float().to(device)

            if args.data_type in ['Wang_CVPR20', 'Ojha_CVPR23']: 
                if args.model_name=='clip':
                    outputs = model(inputs, return_all=True).view(-1).unsqueeze(1)
                else:
                    outputs = model(inputs).view(-1).unsqueeze(1)

                y_pred.extend(outputs.sigmoid().detach().cpu().numpy())
                y_true.extend(labels.cpu().numpy())
            else:
                outputs = model(inputs)
                y_pred.extend(outputs.detach().cpu().numpy())
                y_true.extend(labels.cpu().numpy())

    if args.conv_features and dist.get_rank() == 0:
        from artifacts import extraction_tensor
        import re

        # Combine features for each block across all batches and save
        for stage, blocks in features_dict.items():
            for block, features in blocks.items():
                combined_features = torch.cat(features, dim=0)
                extraction_tensor(combined_features, "./artifacts/resnet50", f'stage_{stage}_block_{block}', device=device, print_scalars=True)
                # print(f"stage_{stage}_block_{block} | shape: {combined_features.shape}")

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    if data_type == 'ImageNet_mini':
        import scipy
        probs = scipy.special.softmax(y_pred, axis=1)
        y_pred_ac = np.argmax(probs, axis=1)
        acc = accuracy_score(y_true, y_pred_ac)
        ap = 0
        auc = 0
    else:
        # General metrics
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

    if dist.get_rank() == 0:
        print(f'Average Precision: {ap}')
        print(f'Accuracy: {acc}')
        print(f'ROC AUC Score: {auc}')

    # Convert predictions to binary labels based on a threshold
    y_pred_binary = (y_pred > 0.5).astype(int)

    if per_class_metrics:
        # Compute per-class metrics and store in a dictionary
        per_class_results = {}
        for class_label in np.unique(y_true):
            precision = precision_score(y_true, y_pred_binary, pos_label=class_label, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, pos_label=class_label, zero_division=0)
            f1 = f1_score(y_true, y_pred_binary, pos_label=class_label, zero_division=0)

            per_class_results[f'class_{int(class_label)}'] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
            }

        # Return a structured dictionary with overall and per-class metrics
        return {'overall': {'accuracy': acc, 'average_precision': ap, 'auc': auc}, 'per_class': per_class_results}
    else:
        # Return metrics in the original format if per_class_metrics is False
        return ap, acc, auc


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
    if args is not None and args.model_name == 'clip':
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
    if args is not None and args.model_name == 'clip':
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
    if args is not None and args.model_name == 'clip':
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

def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):
    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def measure_global_sparsity(
    model, 
    weight = True,
    bias = False, 
    conv2d_use_mask = False,
    linear_use_mask = False
    ):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity
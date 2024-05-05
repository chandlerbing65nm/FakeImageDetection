
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
import os

from commons import get_model_flops, get_model_sparsity, get_modules, get_weights, normalize_scores, count_unmasked_weights, compute_erks

os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'

def prune_layers(model, layer_name, layer_module, pruning_method, amount):
    current_layer = dict(model.named_modules())[layer_name]
    pruning_method(current_layer, name="weight", amount=amount)

def evaluate_pruned_model(model, data_loader, device, args=None):
    model.eval()
    total_samples = len(data_loader.dataset)
    running_loss = 0.0
    y_true, y_pred = [], []
    
    for batch_data in data_loader:
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
    return acc, ap

def compute_lamp_amounts(model, amount):
    """
    Compute normalization schemes.
    """
    unmaskeds = count_unmasked_weights(model)
    num_surv = int(np.round(unmaskeds.sum() * (1.0 - amount)))

    flattened_scores = [normalize_scores(w**2).view(-1) for w in get_weights(model)]
    concat_scores = torch.cat(flattened_scores, dim=0)
    topks, _ = torch.topk(concat_scores, num_surv)
    threshold = topks[-1]

    # We don't care much about tiebreakers, for now.
    final_survs = [
        torch.ge(score, threshold * torch.ones(score.size()).to(score.device)).sum() for score in flattened_scores
    ]
    amounts = []
    for idx, final_surv in enumerate(final_survs):
        amounts.append(1.0 - (final_surv / unmaskeds[idx]))

    return amounts

def compute_erk_amounts(model, amount):

    unmaskeds = count_unmasked_weights(model)
    ers = compute_erks(model)

    num_layers = ers.size(0)
    layers_to_keep_dense = torch.zeros(num_layers)
    total_to_survive = (1.0 - amount) * unmaskeds.sum()  # Total to keep.

    # Determine some layers to keep dense.
    is_eps_invalid = True
    while is_eps_invalid:
        unmasked_among_prunables = (unmaskeds * (1 - layers_to_keep_dense)).sum()
        to_survive_among_prunables = total_to_survive - (layers_to_keep_dense * unmaskeds).sum()

        ers_of_prunables = ers * (1.0 - layers_to_keep_dense)
        survs_of_prunables = torch.round(to_survive_among_prunables * ers_of_prunables / ers_of_prunables.sum())

        layer_to_make_dense = -1
        max_ratio = 1.0
        for idx in range(num_layers):
            if layers_to_keep_dense[idx] == 0:
                if survs_of_prunables[idx] / unmaskeds[idx] > max_ratio:
                    layer_to_make_dense = idx
                    max_ratio = survs_of_prunables[idx] / unmaskeds[idx]

        if layer_to_make_dense == -1:
            is_eps_invalid = False
        else:
            layers_to_keep_dense[layer_to_make_dense] = 1

    amounts = torch.zeros(num_layers)

    for idx in range(num_layers):
        if layers_to_keep_dense[idx] == 1:
            amounts[idx] = 0.0
        else:
            amounts[idx] = 1.0 - (survs_of_prunables[idx] / unmaskeds[idx])
    return amounts

def iterative_pruning_finetuning(
    model, 
    criterion, 
    optimizer, 
    scheduler,
    train_loader, 
    val_loader, 
    mask_loader,
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
            print(f'\nPruninf Iteration #{i}\n')

        torch.distributed.barrier() # Synchronize all processes

        if args.pruning_test:
            original_model = copy.deepcopy(model)  
            conv2d_layers = [(name, module) for name, module in original_model.named_modules() if isinstance(module, torch.nn.Conv2d) and "downsample" not in name]

            amount = args.conv2d_prune_amount

            for (layer_name, layer_module) in conv2d_layers:
                model = copy.deepcopy(original_model) 
                prune_layers(model, layer_name, layer_module, torch.nn.utils.prune.l1_unstructured, amount)
                acc, ap = evaluate_pruned_model(model, val_loader, device, args=args)
                del model

                if torch.distributed.get_rank() == 0:
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

        elif args.pruning_test_ft:
            current_ap_scores = []
            current_og_ap_scores = []
            perf_amounts = []
            prune_amounts = [] 

            if i == 0:
                og_ap_scores = []
                ap_scores = []
                conv2d_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d) and "downsample" not in name]

            amount = args.conv2d_prune_amount

            # computing the sensitivity with pruning
            for (layer_name, layer_module) in tqdm(conv2d_layers, f"{args.mask_type} masked by {args.ratio}% in {args.band} band and pruned {amount*100:.2f}% layerwise"):
                prune_model = copy.deepcopy(model) 
                prune_layers(prune_model, layer_name, layer_module, torch.nn.utils.prune.l1_unstructured, amount)
                acc, ap = evaluate_pruned_model(prune_model, mask_loader, device, args=args)
                current_ap_scores.append(ap)
                del prune_model
            # Adding new AP scores to previous AP scores element-wise
            if i == 0:
                ap_scores = current_ap_scores
            else:
                ap_scores = [(ap_scores[j] * i + current_ap_scores[j]) / (i + 1) for j in range(len(ap_scores))]


            # computing the sensitivity without pruning
            for (layer_name, layer_module) in tqdm(conv2d_layers, f"{args.mask_type} masked by {args.ratio}% in {args.band} band and pruned 0% layerwise"):
                prune_model = copy.deepcopy(model) 
                prune_layers(prune_model, layer_name, layer_module, torch.nn.utils.prune.l1_unstructured, 0.0)
                acc, ap = evaluate_pruned_model(prune_model, mask_loader, device, args=args)
                current_og_ap_scores.append(ap)
                del prune_model
            # Adding new AP scores to previous AP scores element-wise
            if i == 0:
                og_ap_scores = current_og_ap_scores
            else:
                og_ap_scores = [(og_ap_scores[j] * i + current_og_ap_scores[j]) / (i + 1) for j in range(len(og_ap_scores))]
            
            assert len(og_ap_scores) == len(ap_scores)
            for score in range(len(og_ap_scores)):
                delta_ap = og_ap_scores[score] - ap_scores[score]  # Change in AP
                relative_delta_ap = max(0, delta_ap / og_ap_scores[score])

                # Calculate pruning amount (with offset)
                perf_amount = 1 / (relative_delta_ap * 100 + 1.00001)  # Add a small offset like 1e-5
                perf_amount = min(perf_amount, 1.0)  # Enforce maximum of 1
                perf_amounts.append(perf_amount)

            if args.pruning_method == 'ours_lamp':
                tensor_amounts = compute_lamp_amounts(model, amount)
                ours_method_amounts = [t.item() for t in tensor_amounts]
            elif args.pruning_method == 'ours_erk':
                tensor_amounts = compute_erk_amounts(model, amount)
                ours_method_amounts = [t.item() for t in tensor_amounts]
            elif args.pruning_method == 'ours' or args.pruning_method == 'ours_nomask':
                ours_method_amounts = perf_amounts
            else:
                raise ValueError("invalid pruning method")

            assert len(ours_method_amounts) == len(perf_amounts)
            prune_amounts = [a * b for a, b in zip(ours_method_amounts, perf_amounts)] if args.pruning_method != 'ours' else perf_amounts
            print(prune_amounts)

            assert len(conv2d_layers) == len(prune_amounts)
            for (layer_name, layer_module), amount in zip(conv2d_layers, prune_amounts):
                prune_layers(model, layer_name, layer_module, torch.nn.utils.prune.l1_unstructured, amount)

            model = train_model(
                model, criterion, optimizer, scheduler, 
                train_loader, val_loader, num_epochs=num_epochs_per_pruning, 
                resume_epoch=0, save_path=save_path, early_stopping=None, 
                device=device, args=args, pruning_round=i
                )

            # model = remove_parameters(model)

            if dist.get_rank() == 0:
                print(f'\n\nSparsity: {get_model_sparsity(model) * 100:.2f} (%)')
                print(f"Remaining FLOPs: {get_model_flops(model) * 100:.2f} (%)\n\n")

        torch.distributed.barrier() # Synchronize all processes

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
    pruning_round=None
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
        if 'clip' in args.model_name and not features_exist and args.clip_grad == False:
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
        if 'clip' in args.model_name and features_exist and epoch == resume_epoch and args.clip_grad == False:
            train_loader = load_features("./clip_train_" + features_path, batch_size=args.batch_size, shuffle=False)
            val_loader = load_features("./clip_val_" + features_path, batch_size=args.batch_size, shuffle=False)

            # Assuming files can be safely deleted after loading
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
                    if 'clip' in args.model_name and args.clip_grad == True:
                        outputs = model(batch_inputs, return_all=True).view(-1).unsqueeze(1)
                    else:
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
                if args.pruning_test_ft==False:
                    early_stopping(acc, model, optimizer, epoch)  # Pass the accuracy instead of loss
                    if early_stopping.early_stop:
                        if dist.get_rank() == 0:
                            print("Early stopping")
                        return model
            else:
                if args.pruning_test_ft:
                    scheduler.step()
        
        if dist.get_rank() == 0:
            model_new = copy.deepcopy(model)
            model_new = remove_parameters(model_new)
            state = {
                'epoch': epoch,
                'model_state_dict': model_new.state_dict(),
            }
            save_dir=f"./checkpoints/pruning/{args.model_name.lower()}/{args.pruning_method}"
            os.makedirs(save_dir, exist_ok=True)
            model_save_path = os.path.join(save_dir, f"ep{epoch}_rnd{pruning_round}.pth")
            torch.save(state, model_save_path)

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

    test_opt = {
        'rz_interp': ['bilinear'],
        'loadSize': 256,
        'blur_prob': 0.1,  # Set your value
        'blur_sig': [(0.0 + 3.0) / 2],
        'jpg_prob': 0.1,  # Set your value
        'jpg_method': ['pil'],
        'jpg_qual': [int((30 + 100) / 2)]
    }

    test_transform = test_augment(ImageAugmentor(test_opt), None, args)

    # if data_type == 'GenImage':
    #     test_dataset = GenImage(dataset_path, transform=test_transform)
    # elif data_type == 'Wang_CVPR20' :
    #     test_dataset = Wang_CVPR20(dataset_path, transform=test_transform)
    # elif data_type == 'Ojha_CVPR23' :
    #     test_dataset = OjhaCVPR23(dataset_path, transform=test_transform)
    # else:
    #     raise ValueError("wrong dataset input")

    if 'Ojha_CVPR2023' in dataset_path:
        test_dataset = OjhaCVPR23(dataset_path, transform=test_transform)
    elif 'Wang_CVPR2020' in dataset_path:
        test_dataset = Wang_CVPR20(dataset_path, transform=test_transform)
    else:
        raise ValueError("wrong dataset input")

    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=4)

    if model_name == 'rn50':
        # model = torchvision.models.resnet50(pretrained=pretrained)
        # model.fc = nn.Linear(model.fc.in_features, 1)
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'rn50_mod':
        model = _resnet50(pretrained=False, stride0=1)
        model.fc = ChannelLinear(model.fc.in_features, 1)
    elif model_name.startswith('ViT'):
        model_variant = model_name.split('_')[1] # Assuming the model name is like 'ViT_base_patch16_224'
        model = timm.create_model(model_variant, pretrained=pretrained)
    elif model_name == 'clip_vitl14':
        clip_model_name = 'ViT-L/14'
        model = CLIPModel(clip_model_name, num_classes=1)
    elif model_name == 'clip_rn50':
        clip_model_name = 'rn50'
        model = CLIPModel(clip_model_name, num_classes=1)
    else:
        raise ValueError(f"Model {model_name} not recognized!")

    model = model.to(device)
    model = DistributedDataParallel(model, find_unused_parameters=True)

    checkpoint = torch.load(checkpoint_path)

    # if 'clip' in args.model_name and args.other_model != True and args.clip_ft == False:
    #     model.module.fc.load_state_dict(checkpoint['model_state_dict'])
    if args.other_model:
        model.module.fc.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval() 

    # Calculate FLOPs
    if dist.get_rank() == 0 and dataset_path.endswith(('progan', 'guided')):
        print(f"\n\nRemained FLOPs: {get_model_flops(model) * 100:.2f}")
        print(f'Sparsity: {get_model_sparsity(model) * 100:.2f}\n\n')

    y_true, y_pred = [], []

    disable_tqdm = dist.get_rank() != 0
    data_loader_with_tqdm = tqdm(test_dataloader, f"testing: {dataset_path}", disable=disable_tqdm)

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

    return ap, acc, auc, model

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

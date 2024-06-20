
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

from commons import *
from activations import RunningStats, ActivationExtractor, _get_module_by_name
from RD_PRUNE.tools import *

os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'


def iterative_pruning_finetuning(
    model, 
    calib_model,
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
    rd_dict=None,
    ):
    
    '''
    args.pruning_rounds - number of pruning rounds
    num_epochs_per_pruning - number of epochs per pruning round
    '''

    for i in range(1, args.pruning_rounds + 1):
        if dist.get_rank() == 0:
            print(f'\nPruning Iteration #{i}\n')

        torch.distributed.barrier() # Synchronize all processes

        if args.pruning_test:
            original_model = copy.deepcopy(model)  
            prune_layers = [(name, module) for name, module in original_model.named_modules() if isinstance(module, torch.nn.Conv2d) and "downsample" not in name]

            amount = args.calib_sparsity

            for layer_loop, (layer_name, layer_module) in enumerate(prune_layers):
                model = copy.deepcopy(original_model) 
                pruning_technique(model, layer_name, layer_module, torch.nn.utils.prune.l1_unstructured, amount)
                acc, ap = evaluate_pruned_model(model, mask_loader, device, args=args)
                del model

                if torch.distributed.get_rank() == 0:
                    print(f'Layer {layer_name} -> Acc: {acc:.4f} AP: {ap:.4f}')

                    ##################### Evaluate Model on Each Dataloader ####################
                    with open("pruning_results.txt", "a") as file:  # Open file in append mode
                        if layer_loop == 0:
                            # Write a header at the start of the file if this is the first index
                            file.write("\n\n" + "-" * 28 + "\n")
                            file.write(f"mAP of {args.band} band {args.mask_type} masking ({args.ratio}%) of {args.model_name} pruned at {args.calib_sparsity * 100:.1f}% --> ImageNet weights?: {args.pretrained}\n")
                            if args.pretrained == False: file.write(f"Weights loaded from: {args.checkpoint_path}\n")
                            file.write(f"Dataset: {args.dataset}\n")
                        file.write(f"{ap:.4f}" + "\n")  # Write each result immediately to the file

                    print("##########################\n")

        elif args.pruning_test_ft:
            current_ap_scores = []
            current_og_ap_scores = []
            perf_amounts = []
            prune_amounts = [] 

            if i == 1:
                og_ap_scores = []
                ap_scores = []
                prune_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d) and "downsample" not in name]

            amount = args.calib_sparsity

            # if 'ours' in args.pruning_method:
            #     # Initial sensitivity computation
            #     for (layer_name, layer_module) in tqdm(prune_layers, f"{args.ratio}% {args.mask_type} masking in {args.band} band and pruned 0% layerwise"):
            #         prune_model = copy.deepcopy(model)
            #         pruning_technique(prune_model, layer_name, layer_module, torch.nn.utils.prune.l1_unstructured, 0.0)
            #         acc, ap = evaluate_pruned_model(prune_model, mask_loader, device, args=args)
            #         current_og_ap_scores.append(ap)
            #         del prune_model
            #     if i == 1:
            #         og_ap_scores = current_og_ap_scores
            #     else:
            #         og_ap_scores = [(og_ap_scores[j] * i + current_og_ap_scores[j]) / (i + 1) for j in range(len(og_ap_scores))]

            #     # Sensitivity computation with pruning
            #     for (layer_name, layer_module) in tqdm(prune_layers, f"{args.ratio}% {args.mask_type} masking in {args.band} band and pruned {amount*100:.2f}% layerwise"):
            #         prune_model = copy.deepcopy(model)
            #         pruning_technique(prune_model, layer_name, layer_module, torch.nn.utils.prune.l1_unstructured, amount)
            #         acc, ap = evaluate_pruned_model(prune_model, mask_loader, device, args=args)
            #         current_ap_scores.append(ap)
            #         del prune_model
            #     if i == 1:
            #         ap_scores = current_ap_scores
            #     else:
            #         ap_scores = [(ap_scores[j] * i + current_ap_scores[j]) / (i + 1) for j in range(len(ap_scores))]

            #     for score in range(len(og_ap_scores)):
            #         delta_ap = og_ap_scores[score] - ap_scores[score]  # Change in AP
            #         relative_delta_ap = max(0, delta_ap / og_ap_scores[score])

            #         # Calculate pruning amount (with offset)
            #         perf_amount = 1 / (relative_delta_ap * 100 + 1.00001)  # Add a small offset like 1e-5
            #         perf_amount = min(perf_amount, 1.0)  # Enforce maximum of 1
            #         perf_amounts.append(perf_amount)

            #     if args.pruning_method == 'ours_lamp':
            #         tensor_amounts = compute_lamp_amounts(model, amount)
            #         prune_amounts = [t.item() for t in tensor_amounts]
            #     elif args.pruning_method == 'ours_erk':
            #         tensor_amounts = compute_erk_amounts(model, amount)
            #         prune_amounts = [t.item() for t in tensor_amounts]
            #     elif args.pruning_method == 'ours_rd':
            #         args.worst_case_curve = True
            #         args.synth_data = False
            #         container = rd_dict['container']
            #         calib_loader = rd_dict['calib_loader']
            #         rd_pruner = weight_pruner_loader('rd')
            #         tensor_amounts, target_sparsity = rd_pruner(model, amount, args, calib_loader, container)
            #         prune_amounts = [t.item() for t in tensor_amounts]
            #     elif args.pruning_method == 'ours' or args.pruning_method == 'ours_nomask':
            #         prune_amounts = prune_amounts
            #     else:
            #         raise ValueError("invalid pruning method")

            #     assert len(prune_amounts) == len(perf_amounts)
            #     final_prune_amounts = [a * b for a, b in zip(prune_amounts, perf_amounts)] if args.pruning_method != 'ours' else perf_amounts
            #     print(prune_amounts)
            #     print(perf_amounts)
            #     print(final_prune_amounts)

            # elif args.pruning_method == 'lamp_erk':
            #     tensor_amounts_1 = compute_erk_amounts(model, args.desired_sparsity)
            #     tensor_amounts_2 = compute_lamp_amounts(model, args.calib_sparsity)
            #     prune_amounts_1 = [t.item() for t in tensor_amounts_1]
            #     prune_amounts_2 = [t.item() for t in tensor_amounts_2]
            #     assert len(prune_amounts_1) == len(prune_layers)
            #     assert len(prune_amounts_2) == len(prune_layers)
            #     final_prune_amounts = [a * b for a, b in zip(prune_amounts_1, prune_amounts_2)]
            #     print(final_prune_amounts)

            if args.pruning_method == 'rd':
                args.worst_case_curve = True
                args.synth_data = False
                container = rd_dict['container']
                calib_loader = rd_dict['calib_loader']
                rd_pruner = weight_pruner_loader(args.pruning_method)
                tensor_amounts, target_sparsity = rd_pruner(model, args.desired_sparsity, args, calib_loader, container)
                final_prune_amounts = [t.item() for t in tensor_amounts]
                print(final_prune_amounts)
            else:
                raise ValueError("invalid pruning method")

            assert len(prune_layers) == len(final_prune_amounts)
            for (layer_name, layer_module), amount in zip(prune_layers, final_prune_amounts):
                pruning_technique(model, layer_name, layer_module, torch.nn.utils.prune.l1_unstructured, amount)

            if 'rd' in args.pruning_method:
                os.makedirs(f"./output_files/masks/{args.model_name}/", exist_ok=True)
                mask_save_path = f"./output_files/masks/{args.model_name}/maskratio{args.ratio}_sp{target_sparsity}_{args.model_name}_ndz_{args.maxsps:04d}_rdcurves_mask.pt"
                to_save = {k: v for k, v in model.state_dict().items() if "weight_mask" in k}
                torch.save(to_save, mask_save_path)
                args.iter_start += 1

            model = train_model(
                model, criterion, optimizer, scheduler, 
                train_loader, val_loader, num_epochs=num_epochs_per_pruning, 
                resume_epoch=0, save_path=save_path, early_stopping=None, 
                device=device, pruning_round=i, args=args,
                )

            if dist.get_rank() == 0:
                print(f'\n\nSparsity: {get_model_sparsity(model) * 100:.2f} (%)')
                print(f"Remaining FLOPs: {get_model_flops(model) * 100:.2f} (%)\n\n")

        torch.distributed.barrier()  # Synchronize all processes

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
    pruning_round=None,
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
        if 'clip' in args.model_name and not features_exist and args.trainable_clip == False:
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
        if 'clip' in args.model_name and features_exist and epoch == resume_epoch and args.trainable_clip == False:
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

            for batch_data in tqdm(data_loader, f"{phase}", disable=dist.get_rank() != 0):
                batch_inputs, batch_labels = batch_data
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.float().to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Training'):
                    if 'clip' in args.model_name and args.trainable_clip == True:
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
                    scheduler.step()  # early stopping don't need scheduler
        
        if dist.get_rank() == 0 and pruning_round is not None:
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

    # mask_generator = FrequencyMaskGenerator(ratio=0.15, band='all', transform_type='fourier')
    # mask_generator = PatchMaskGenerator(ratio=0.50)
    # mask_generator = PixelMaskGenerator(ratio=0.50)
    mask_generator = None
    test_transform = test_augment(ImageAugmentor(test_opt), mask_generator, args)

    if 'Ojha_CVPR2023' in dataset_path:
        test_dataset = Ojha_CVPR23(dataset_path, transform=test_transform)
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

    if args.get_activations:
        # Identify the first/last four Conv2D layers to extract activations from
        # layer_names = [
        #     'module.conv1',
        #     'module.layer1.0.conv1',
        #     'module.layer1.0.conv2',
        #     'module.layer1.0.conv3',
        # ]

        layer_names = [
            'module.layer4.1.conv3',
            'module.layer4.2.conv1',
            'module.layer4.2.conv2',
            'module.layer4.2.conv3',
        ]

        # Create activation extractors for each Conv2D layer
        extractors = [ActivationExtractor() for _ in layer_names]

        # Register hooks for each Conv2D layer
        hooks = [_get_module_by_name(model, name).register_forward_hook(extractor.hook) for name, extractor in zip(layer_names, extractors)]

    accumulation_steps = 10
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader_with_tqdm):
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            if 'clip' in args.model_name:
                outputs = model(inputs, return_all=True).view(-1).unsqueeze(1)
            else:
                outputs = model(inputs).view(-1).unsqueeze(1)
            y_pred.extend(outputs.sigmoid().detach().cpu().numpy())
            y_true.extend(labels.cpu().numpy())

            if args.get_activations:
                # Periodically update the running statistics
                if (i + 1) % accumulation_steps == 0:
                    for extractor in extractors:
                        extractor.update_stats()

    if args.get_activations:
        # Update the running statistics one last time for any remaining activations
        for extractor in extractors:
            extractor.update_stats()

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    if dist.get_rank() == 0:
        print(f'Average Precision: {ap}')
        print(f'Accuracy: {acc}')
        print(f'ROC AUC Score: {auc}')

    if args.get_activations:
        # Collect final statistics for each layer
        average_activations = [extractor.stats.mean for extractor in extractors]

        # Clean up hooks
        for hook in hooks:
            hook.remove()

        return average_activations, ap, acc, auc, model
    else:
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
        # transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),)
    return transforms.Compose(transform_list)

def test_augment(augmentor, mask_generator=None, args=None):
    transform_list = []
    if mask_generator is not None:
        transform_list.append(transforms.Lambda(lambda img: mask_generator.transform(img)))
    transform_list.extend([
        # transforms.Lambda(lambda img: img.rotate(27)), # Add rotation - 9, 18, 27,
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),  # Add random translation
        # transforms.Lambda(lambda img: augmentor.custom_resize(img)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

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

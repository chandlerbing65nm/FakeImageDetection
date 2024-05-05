# this code os from https://github.com/Akimoto-Cris/RD_PRUNE.git

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, LayerNorm
import warnings

warnings.simplefilter("ignore", UserWarning)

device = torch.device("cuda:0")

PARAMETRIZED_MODULE_TYPES = (torch.nn.Linear, torch.nn.Conv2d,)
NORM_MODULE_TYPES = (torch.nn.BatchNorm2d, torch.nn.LayerNorm)

def compute_erks(model):
    wlist = get_weights(model)
    erks = torch.zeros(len(wlist))
    for idx, w in enumerate(wlist):
        if w.dim() == 4:
            erks[idx] = w.size(0) + w.size(1) + w.size(2) + w.size(3)
        else:
            erks[idx] = w.size(0) + w.size(1)
    return erks

def normalize_scores(scores):
    """
    Normalizing scheme for LAMP.
    """
    # sort scores in an ascending order
    sorted_scores, sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape, device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[: len(scores_cumsum_temp) - 1]
    # normalize by cumulative sum
    sorted_scores /= scores.sum() - scores_cumsum
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape, device=scores.device)
    new_scores[sorted_idx] = sorted_scores

    return new_scores.view(scores.shape)

@torch.no_grad()
def count_unmasked_weights(model):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    mlist = get_modules(model)
    unmaskeds = []
    for m in mlist:
        unmaskeds.append(m.weight.count_nonzero())
    return torch.FloatTensor(unmaskeds)
    
def get_model_sparsity(model):
    prunables = 0
    nnzs = 0
    for name, m in model.named_modules():
        if _is_prunable_module(m, name):
            prunables += m.weight.data.numel()
            nnzs += m.weight.data.nonzero().size(0)
    return 1 - nnzs/prunables

def _is_prunable_module(m, name=None):
    return isinstance(m, nn.Conv2d) and "downsample" not in name

def get_modules(model):
    modules = []
    for name, m in model.named_modules():
        if _is_prunable_module(m, name):
            modules.append(m)
    return modules

def get_weights(model):
    weights = []
    for name, m in model.named_modules():
        if _is_prunable_module(m, name):
            weights.append(m.weight)
    return weights

def pushattr(layers, container, attr, includenorm, direction, prefix=""):
    if isinstance(getattr(container, attr, None), PARAMETRIZED_MODULE_TYPES) or (
        isinstance(getattr(container, attr, None), NORM_MODULE_TYPES) and includenorm
    ):
        if direction == 0:
            layers[0].append(getattr(container, attr))
        else:
            setattr(container, attr, layers[0][0])
            layers[0] = layers[0][1 : len(layers[0])]


def pushlist(layers, container, attr, includenorm, direction, prefix=""):
    if isinstance(container[attr], PARAMETRIZED_MODULE_TYPES) or (
        isinstance(container[attr], NORM_MODULE_TYPES) and includenorm
    ):
        # container[attr] = TimeWrapper(container[attr], prefix)
        if direction == 0:
            layers[0].append(container[attr])
        else:
            container[attr] = layers[0][0]
            layers[0] = layers[0][1 : len(layers[0])]
    else:
        pushconv(layers, container[attr], includenorm, direction, prefix=prefix)


def pushconv(layers, container, includenorm=True, direction=0, prefix="model"):
    return [m for m in container.modules() if isinstance(m, PARAMETRIZED_MODULE_TYPES)]

def findconv(net, includenorm=True):
    layers = pushconv([[]], net, includenorm)
    return layers

@torch.no_grad()
def _count_unmasked_weights(model):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    mlist = get_modules(model)
    unmaskeds = []
    for m in mlist:
        unmaskeds.append(m.weight.count_nonzero())
    return torch.FloatTensor(unmaskeds)


@torch.no_grad()
def _count_total_weights(model):
    """
    Return a 1-dimensional tensor of #total weights.
    """
    wlist = get_weights(model)
    numels = []
    for w in wlist:
        numels.append(w.numel())
    return torch.FloatTensor(numels)

def hooklayers(layers):
    return [Hook(layer) for layer in layers]

class Hook:
    def __init__(self, module, backward=False):
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input_tensor = input[0]
        self.input = torch.tensor(input[0].shape[1:])
        self.output = torch.tensor(output[0].shape[1:])
        self.input_tensor = input[0]
        self.output_tensor = output[0]

    def close(self):
        self.hook.remove()


@torch.no_grad()
def get_model_flops(net):
    net.eval()

    dummy_input = torch.zeros((1, 3, 224, 224), device=next(net.parameters()).device)

    layers = findconv(net, False)
    unmaskeds = _count_unmasked_weights(net)
    totals = _count_total_weights(net)

    hookedlayers = hooklayers(layers)
    _ = net(dummy_input)
    fil = [hasattr(h, "output") for h in hookedlayers]
    if False in fil:
        layers = [layers[i] for i in range(len(layers)) if fil[i]]
        hookedlayers = [hookedlayers[i] for i in range(len(hookedlayers)) if fil[i]]
        unmaskeds = [unmaskeds[i] for i in range(len(unmaskeds)) if fil[i]]
        totals = [totals[i] for i in range(len(totals)) if fil[i]]

    output_dimens = [hookedlayers[i].output for i in range(0, len(hookedlayers))]
    for l in hookedlayers:
        l.close()

    denom_flops = 0.0
    nom_flops = 0.0

    for o_dim, surv, tot, m in zip(output_dimens, unmaskeds, totals, layers):
        if isinstance(m, torch.nn.Conv2d):
            denom_flops += o_dim[-2:].prod() * tot + (0 if m.bias is None else o_dim.prod())
            nom_flops += o_dim[-2:].prod() * surv + (0 if m.bias is None else o_dim.prod())
        elif isinstance(m, torch.nn.Linear):
            denom_flops += tot + (0 if m.bias is None else o_dim.prod())
            nom_flops += surv + (0 if m.bias is None else o_dim.prod())

    return nom_flops / denom_flops


import argparse  # For handling command-line arguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate FLOPs for a given model')
    parser.add_argument('model', type=str, help='Name or type of the model architecture')
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='Device to use for calculations (default: cuda:0)')

    args = parser.parse_args()

    # Model loading logic
    if args.model == 'resnet18':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)  # Example
    elif args.model == 'densenet121':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)  # Example
    else:
        # Handle cases where the model name is not directly recognized 
        print(f"Model architecture '{args.model}' not supported for automatic loading.")
        exit(1) 

    model.to(args.device)  # Move the model to the specified device

    flops_ratio = get_model_flops(model)
    print(f"Model FLOPs Ratio: {flops_ratio}")
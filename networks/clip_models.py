import clip 
from PIL import Image
import torch.nn as nn
import torch

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, device="cpu", num_classes=1, unfreeze=False):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device=device) # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS[name], num_classes )
 
        for param in self.model.parameters():
            param.requires_grad = freeze

    def forward(self, x, return_feature=False, return_all=False):
        encoded = self.model.encode_image(x)  # Always encode the image
        if return_feature:
            return encoded  # Just return features if requested
        elif return_all:
            return self.fc(encoded)  # Apply the linear layer and return
        else:
            return self.fc(encoded)  # Default to returning the final output

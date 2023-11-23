from .clip import clip 
from PIL import Image
import torch.nn as nn


CHANNELS = {
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS[name], num_classes )
 
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, return_feature=False, return_all=False):
        if return_feature:
            return self.model.encode_image(x) 
        elif return_all:
            return self.fc(self.model.encode_image(x))
        else:
            return self.fc(x)

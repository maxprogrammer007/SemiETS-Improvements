# models/backbone.py

import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self, name="resnet18", pretrained=True, freeze=True):
        super().__init__()

        if name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            self.out_channels = 512
            self.backbone = nn.Sequential(*list(model.children())[:-2])
        else:
            raise ValueError("Unsupported backbone")

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: (B, 3, H, W)
        return: feature map (B, C, H', W')
        """
        return self.backbone(x)

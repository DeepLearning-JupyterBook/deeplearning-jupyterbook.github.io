"""
A module like "models.py" often contain a list of all models used in a project.

In this example we only have one model LinearProbe.

If this module becomes too completed (too many functions) it can be broken down into smaller modules
corresponding to different architectures or blocks.
"""

import torch
from torchvision import models


class LinearProbe(torch.nn.Module):
    def __init__(self, device, pretrained=None, feature_size=None):
        super(LinearProbe, self).__init__()

        if pretrained is None:
            pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
            pretrained = torch.nn.Sequential(*list(pretrained.children())[:8])
            pretrained.eval()
            feature_size = 512 * 7 * 7

        self.feature_extractor = pretrained
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        self.fc = torch.nn.Linear(2 * feature_size, 2)

    def forward(self, x1, x2):
        x1 = self.feature_extractor(x1)
        x1 = torch.flatten(x1, start_dim=1)

        x2 = self.feature_extractor(x2)
        x2 = torch.flatten(x2, start_dim=1)

        x = torch.cat([x1, x2], dim=1)
        return self.fc(x)

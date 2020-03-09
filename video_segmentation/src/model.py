
import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

import config

resnet18 = models.resnet18(pretrained=True)

modules=list(resnet18.children())[:-1]
resnet18=nn.Sequential(*modules)

for param in resnet18.parameters():
    param.requires_grad = False

resnet18.to(config.device)
resnet18.eval()

# summary(resnet18, input_size=(3, 224, 224))

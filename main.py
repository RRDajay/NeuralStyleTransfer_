import torch
import torch.optim as optim
import torchvision 
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from utilityfunctions import image_to_tensor, tensor_to_image

# Load model
model = torchvision.models.vgg19(pretrained=True, progress=True)

# Freeze weights
for param in model.parameters():
    param.requires_grad_(False)

# Change maxpool2d layers to avgpool2d
for i, layer in enumerate(model.features):
    if isinstance(layer, torch.nn.MaxPool2d):
        model.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

# Send model to gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device).eval()


print(model.classifier)












import torch
import torch.nn as nn

import numpy as np

import config
from model import resnet18
from data_loader import dataloaders

def extract_features(model):
    all_img_features = []
    for img in dataloaders['test']:
        img = img.type(torch.FloatTensor).to(device)
        features = model(img)
        features = features.squeeze(2).squeeze(2).detach().cpu().numpy()
        for feature in features:
            all_img_features.append(feature)
        
    np.save(config.output_path, np.array(all_img_features))

if __name__ == '__main__':
    device = config.device
    model = resnet18.to(device)

    extract_features(model)
from collections import defaultdict
import time 
import copy 

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import config
from model import UNet
from data_loader import dataloaders

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.FloatTensor).to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                torch.save(model.state_dict(), config.model_name)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    return model

def visualize_results(model):
    model.eval()    # Set model to evaluate mode
    for inputs, labels in dataloaders['val']:
        inputs = inputs.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor).to(device)
        model_outputs = model(inputs)
        for b in range(len(model_outputs)):
            fig, (image1, image2, image3) = plt.subplots(1, 3)

            original = cv2.normalize(np.moveaxis(inputs[b].cpu().numpy(), 0, 2), None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            image1.title.set_text('Original image')
            image1.imshow(original)

            image2.title.set_text('Ground truth mask')
            image2.imshow(np.moveaxis(labels[b].cpu().numpy(), 0, 2))

            image3.title.set_text('Predicted mask')
            image3.imshow(np.moveaxis(model_outputs[b].detach().cpu().numpy(), 0, 2))

            plt.show()

if __name__ == '__main__':
    device = config.device
    num_class = config.num_class

    model = UNet(num_class).to(device)

    if(config.train):
        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)
        model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=40)
    else:
        model.load_state_dict(torch.load(config.model_name))
        visualize_results(model)
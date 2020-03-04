import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torchvision.utils

import config

class MRIDataset(Dataset):
    def __init__(self, input_path, mask_path):
        self.input_images = np.load(input_path)
        self.target_masks = np.load(mask_path)

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        return [np.moveaxis(image, 2, 0), np.moveaxis(mask, 2, 0)]

train_set = MRIDataset(input_path=config.train_input_image_path, mask_path=config.train_mask_image_path)
val_set = MRIDataset(input_path=config.test_input_image_path, mask_path=config.test_mask_image_path)

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = config.batch_size

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

if __name__ == '__main__':
    # Get a batch of training data
    inputs, masks = next(iter(dataloaders['train']))
    inputs.to(config.device)
    masks.to(config.device)
    print(inputs.type, masks.type)
    print(inputs.shape, masks.shape)

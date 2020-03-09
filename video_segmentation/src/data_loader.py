import numpy as np
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torchvision.utils
import torchvision.transforms as transforms

import PIL
import config

class BrainSurgeryDataset(Dataset):
    def __init__(self, image_path):
        self.images = np.load(image_path)
        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # print(image)
        # print(image.shape)
        # image = cv.resize(image, dsize=(224,224), interpolation=cv.INTER_CUBIC)
        image = PIL.Image.fromarray(image)
        image = self.normalize(self.to_tensor(self.scaler(image)))
        image = np.array(image)
        # print(image)
        # print(image.shape)
        # quit()
        return image # [3,224,224]

test_set = BrainSurgeryDataset(image_path=config.input_image_path)

image_datasets = {
    'test': test_set
}

batch_size = config.batch_size

dataloaders = {
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, ),
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

if __name__ == '__main__':
    # Get a batch of training data
    inputs = next(iter(dataloaders['test']))
    inputs.to(config.device)
    print(inputs.type)
    print(inputs.shape)

import torch

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_class = 3 # do not change
model_name = '../models/mri.pt'
train_input_image_path = '../data/images/train_images.npy'
train_mask_image_path = '../data/masks/train_masks.npy'
test_input_image_path = '../data/images/test_images.npy'
test_mask_image_path = '../data/masks/test_masks.npy'

train = False

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
input_image_path = '../data/Kanniya/Segment01.npy'
output_path = '../data/Kanniya/feature_Segment01.npy'
train = False
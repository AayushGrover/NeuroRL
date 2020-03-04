import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# file_names = range(3064)
# train_images = []
# train_masks = []
# test_images = []
# test_masks = []

# for file_name in file_names:
#     img = cv2.imread('../data/images/'+str(file_name)+'.png')
#     mask = cv2.imread('../data/masks/'+str(file_name)+'.png')
#     norm_mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     if(file_name < 3000):
#         train_images.append(img)
#         train_masks.append(norm_mask)
#     else:
#         test_images.append(img)
#         test_masks.append(norm_mask)

# np.save('train_images.npy', np.array(train_images))
# np.save('train_masks.npy', np.array(train_masks))

# np.save('test_images.npy', np.array(test_images))
# np.save('test_masks.npy', np.array(test_masks))

images = np.load('test_images.npy')
print(images.shape)

masks = np.load('test_masks.npy')
print(masks.shape)
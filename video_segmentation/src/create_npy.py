import numpy as np
from glob import glob
import cv2 as cv

input_path = '../data/Kanniya/Segment01/*'
output_path = '../data/Kanniya/Segment01'

l = []
for img in glob(input_path):
    np_img = cv.imread(img)
    l.append(np_img)

l = np.array(l)
np.save(output_path, l)
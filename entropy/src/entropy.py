# load() function  
  
import numpy as np 
from scipy.stats import entropy
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
b = np.load('feature_Segment01.npy') 

#testing entropy function for pairwise frames
l = list()
for i in range(600):
  l.append(entropy(b[i],b[i+1]))

print (l)



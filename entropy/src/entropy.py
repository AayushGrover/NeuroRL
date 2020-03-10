# load() function  
  
import numpy as np 
from scipy.stats import entropy
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
b = np.load('feature_Segment01.npy') 


def entropy1(labels, base=None): 
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)


#testing entropy function
l1 = [1,0]
l2 = [0,1]
print (entropy(l1,l2))




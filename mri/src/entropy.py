import numpy as np 
import math
b = np.load('feature_Segment01.npy') 

entropies = list()


for window_size in range(2,512):
    l = b[0:window_size]
    var = np.var(l,axis = 0)  # Calculate variance
    entropy = list()
    for t in var:
        # If entropy throws an error becuase of zero variance uncomment this code
        if (t == 0):
            entropy.append(0) # In case variance is zero
        else:
            entropy.append(( 1/2 * math.log(2* math.pi * math.e * t *t))) # calculate entropy for each window size
    entropies.append(entropy)
    
L1_norm = list()
for i in range(2,509):
    L1_norm.append(np.linalg.norm(np.asarray(entropies[i])-np.asarray(entropies[i+1])))

print (L1_norm) # This is the L1 norm between 0 : k frame and 0 : k+1 th frame 

# So we can check if the enropy for the k+1 th frame increased or decreased and decide the split accordingly
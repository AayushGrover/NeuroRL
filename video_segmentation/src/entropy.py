import numpy as np 
import math
import matplotlib.pyplot as plt; plt.rcdefaults()

b = np.load('feature_Segment01.npy') 

frames = []
window_size = 20
for i in range(2,490):
    l = b[i:i+20]
    entropy = np.var(l, axis=0)  # Calculate variance
    l1_norm = np.linalg.norm(entropy)
    # print(l1_norm)
    frames.append(l1_norm)
threshold = 0.45
index = list()
for i in range(487):
    if(frames[i+1] - frames[i] > threshold):
        index.append(i) 
print ("Timestamps:")
for i in index:
    print (i/len(frames) * 300) # timestamp of the ith frame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

y_pos = list()
for i in range(488):
    y_pos.append(i)
performance = frames 
plt.xlabel("timestamp windows")
plt.ylabel("Norm of entropy")
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.show()

# So we can check if the enropy for the k+1 th frame increased or decreased and decide the split accordingly
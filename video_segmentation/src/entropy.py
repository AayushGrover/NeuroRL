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
# threshold = 0.45
# index = list()
# for i in range(487):
#     if(frames[i+1] - frames[i] > threshold):
#         index.append(i) 
# print ("Timestamps:")
# for i in index:
#     print (i/len(frames) * 300) # timestamp of the ith frame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

y_pos = list()
for i in range(488):
    y_pos.append(i)
performance = frames 
plt.xlabel("timestamp windows")
plt.ylabel("Norm of entropy")
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.show()

performance_for_top_k = list()
k = 42
for (x, y) in zip(y_pos, performance):
    performance_for_top_k.append((x, y))
performance_for_top_k.sort(key=lambda x: x[1])
top_k_pairs = performance_for_top_k[:k]
top_k_pairs.sort(key=lambda x: x[0])
for (t, val) in top_k_pairs:
    print(f'Frame at which segmentation should happen: {t}, Time at which segmentation should happen: {t/len(frames)*300}')

# So we can check if the enropy for the k+1 th frame increased or decreased and decide the split accordingly

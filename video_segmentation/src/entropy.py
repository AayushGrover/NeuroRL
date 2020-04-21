import numpy as np 
import math
import matplotlib.pyplot as plt; plt.rcdefaults() 
import datetime

def get_entropy(all_frames, window_size=20):
    frames = []
    for i in range(2,490):
        l = all_frames[i:i+window_size]
        entropy = np.var(l, axis=0)  # Calculate variance
        l1_norm = np.linalg.norm(entropy)
        # print(l1_norm)
        frames.append(l1_norm)
    return frames
    # threshold = 0.45
    # index = list()
    # for i in range(487):
    #     if(frames[i+1] - frames[i] > threshold):
    #         index.append(i) 
    # print ("Timestamps:")
    # for i in index:
    #     print (i/len(frames) * 300) # timestamp of the ith frame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

def smoothen(curve, k):
    new_curve = curve
    l = len(curve)

    for i in range(l):
        s = 0
        for j in range(k):
            try:
                s += curve[i+j]
            except:
                continue
        new_curve[int((i+j)/2)] = s/k
    
    return new_curve

def plot(frames, smoothing_factor):
    y_pos = list()
    for i in range(488):
        y_pos.append(i)
    performance = frames 
    smooth_performance = smoothen(curve=performance, k=smoothing_factor)

    plt.xlabel("timestamp windows")
    plt.ylabel("Norm of entropy")
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.plot(y_pos, smooth_performance, 'r')
    plt.show()
    
    return y_pos, smooth_performance

def check(timestamp, seq, l):
    for t,_ in seq:
        if abs((t/l*300)-(timestamp/l*300)) < 2:
            return False
    return True

def get_top_k_performance(y_pos, performance_list, k):
    performance_for_top_k = list()
    for (x, y) in zip(y_pos, performance_list):
        performance_for_top_k.append((x, y))
    performance_for_top_k.sort(key=lambda x: x[1])
    # top_k_pairs = performance_for_top_k[:k]
    top_k_pairs = [performance_for_top_k[0]]
    k_count = 1
    for t,performance in performance_for_top_k:
        if abs(performance - top_k_pairs[-1][1]) > 0.2 and check(t,top_k_pairs,len(performance_list)):
            top_k_pairs.append((t,performance))
            k_count += 1
        if k_count == k:
            break


    top_k_pairs.sort(key=lambda x: x[0])
    for (t, _) in top_k_pairs:
        print(f'Frame at which segmentation should happen: {t}, Time at which segmentation should happen: {datetime.timedelta(seconds=(t/len(performance_list)*300))}')

# So we can check if the enropy for the k+1 th frame increased or decreased and decide the split accordingly

if __name__ == "__main__":
    
    b = np.load('feature_Segment01.npy')
    
    frames = get_entropy(b)
    y_pos, performance = plot(frames, smoothing_factor=3)
    get_top_k_performance(y_pos, performance,k=15)

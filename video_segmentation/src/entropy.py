import numpy as np 
import math
import matplotlib.pyplot as plt; plt.rcdefaults() 
import datetime
from statistics import mean

def get_entropy(all_frames, n, window_size=20):
    frames = []

    for i in range(2,n-window_size-2):
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

def plot(frames, n, smoothing_factor, window_size):
    y_pos = list()
    for i in range(n-window_size-4):
        y_pos.append(i)
    performance = frames 
    smooth_performance = smoothen(curve=performance, k=smoothing_factor)

    plt.xlabel("timestamp windows")
    plt.ylabel("Norm of entropy")
    print(len(y_pos), len(performance))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.plot(y_pos, smooth_performance, 'r')
    plt.show()
    
    return y_pos, smooth_performance

def check(timestamp, seq, l):
    for t,_ in seq:
        if abs((t/l*300)-(timestamp/l*300)) < 5:
            return False
    return True

def get_scores(l, avg_factor=7):
    scores = list()
    n=len(l)
    for (i,_) in l:
        if i-avg_factor >= 0 and i+avg_factor < n:
            window = [perf[1] for perf in l[i-avg_factor+1:i+avg_factor-1]]
            
            e_avg = np.mean(np.array(window))
            e_max = max(window)
            e_min = min(window)
            score = 2*e_avg - l[i-avg_factor][1] - l[i+avg_factor][1] - (e_max - e_min) 
            scores.append((i,score))
        else:
            continue
    return scores

def get_top_k_performance(y_pos, performance_list, k):
    performance_for_top_k = list()
    for (x, y) in zip(y_pos, performance_list):
        performance_for_top_k.append((x, y))

    scores = get_scores(performance_for_top_k)
    scores.sort(key=lambda z: z[1], reverse=True)
    
    performance_for_top_k = scores
    top_k_pairs = [performance_for_top_k[0]]
    k_count = 1
    
    for t,performance in performance_for_top_k:
        if check(t,top_k_pairs,len(performance_list)):
            top_k_pairs.append((t,performance))
            k_count += 1
        else:
            if(abs(performance - top_k_pairs[-1][1]) > 3):
                top_k_pairs.append((t,performance))
                k_count += 1    
        if k_count == k:
            break


    top_k_pairs.sort(key=lambda x: x[0])
    for (t, _) in top_k_pairs:
        print(f'Frame at which segmentation should happen: {t}, Time at which segmentation should happen: {datetime.timedelta(seconds=(t/len(performance_list)*300))}')
        # print(f'Frame at which segmentation should happen: {t}, Time at which segmentation should happen: {(t/len(performance_list)*300)}')

# So we can check if the enropy for the k+1 th frame increased or decreased and decide the split accordingly

if __name__ == "__main__":
    
    b = np.load('../data/Celein/feature_Segment01.npy')
    window_size = 20

    frames = get_entropy(b, n=b.shape[0], window_size=window_size)
    y_pos, performance = plot(frames, n=b.shape[0], smoothing_factor=3, window_size=window_size)
    get_top_k_performance(y_pos, performance,k=10)

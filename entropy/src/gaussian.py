  
import numpy as np 
b = np.load('feature_Segment01.npy') 
print (b[:,0])
mu = list() #list of means of the features
sigma = list() #list of std of the features
for i in range(512):
    mu.append(np.mean(b[:,i]))
    sigma.append(np.std(b[:,i]))


s = np.random.normal(mu[0], sigma[0], 1000) #sampling from the gaussian of the first feature
print (s)
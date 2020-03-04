import numpy as np
import torch

def div_reward(seq):
    '''
    Computes a metric for diversity in a sequence of images
    seq: sequence of images (1, seq_len, dim) 
    '''
    _seq = seq.detach()
    _seq = _seq.squeeze()
    normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
    dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
 
    
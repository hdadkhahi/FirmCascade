
from __future__ import division
import numpy as np

def z_normalize(trX):
    m = np.mean(trX, axis = 0)
    v = np.std(trX, axis = 0)
    trX = (trX - m) / v  
    return trX

def feature_selection(trX, teX, f_subset, f, z):
    
    
    if f == 5:
        trX11 = np.zeros((trX.shape[0], 5))
        add_features = lambda x, y: [x, y, x**2, y**2, x*y]
        for i1 in range(trX.shape[0]):
            trX11[i1, :] = add_features(trX[i1, f_subset[0]], trX[i1, f_subset[1]])
        teX11 = np.zeros((teX.shape[0], 5))
        for i1 in range(teX.shape[0]):
            teX11[i1, :] = add_features(teX[i1, f_subset[0]], teX[i1, f_subset[1]])   
        if z == 1:
            trX11 = z_normalize(trX11) 
            teX11 = z_normalize(teX11) 
    else:
        trX11 = trX[:, f_subset]
        teX11 = teX[:, f_subset]             
    
    return trX11, teX11
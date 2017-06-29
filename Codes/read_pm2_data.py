
import numpy as np
import scipy.io as sio
from sklearn import preprocessing


def read_pm2_data():
    filename = 'puffMarkerdatatime.mat'
    test = sio.loadmat(filename)
    A = test['A']
    P = test['P']
    A = preprocessing.scale(A)
    inds = np.arange(A.shape[0])
    np.random.seed(seed=0)
    np.random.shuffle(inds)
    A = A[inds, :]
    P = P[:, inds]
    print A.shape, P.shape
    print P.sum(axis=1)
    # A is x; P is y
    return A, P.reshape((3836, ))

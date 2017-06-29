
from __future__ import division
import theano
from theano import tensor as T
import numpy as np
#from load import mnist
from sklearn import svm
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams()
#from sklearn import grid_search
#import time
#import matplotlib.pyplot as plt
    
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    np.random.seed(seed=1)
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates
    
def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates
    
def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model0(X,w,b):
    pyx = softmax(T.dot(X, w) + b)
    return pyx

def model00(X,w,b):
    pyx = 1.0 / (1.0 + T.exp(-T.dot(X, w) - b))
    return pyx

def model1(X, w_h, w_o):
    h = T.nnet.sigmoid(T.dot(X, w_h))
    pyx = T.nnet.softmax(T.dot(h, w_o))
    return pyx

def model2(X, w_h1, w_o, b1,bo,p,s):
    h = dropout(rectify(b1+T.dot(X, s*w_h1)),p)
    #h2 = dropout(rectify(b2+T.dot(h1, s*w_h2)),p)
    pyx = softmax(bo+T.dot(h, w_o))
    return pyx

def model(X, w_h1, w_h2, w_o, b1,b2,bo,p,s):
    h1 = dropout(rectify(b1+T.dot(X, s*w_h1)),p)
    h2 = dropout(rectify(b2+T.dot(h1, s*w_h2)),p)
    pyx = 1.0 / (1.0 + T.exp(-T.dot(h2, w_o) - bo))
    return pyx

def model3(X, w_h1, w_o, b1,bo,p,s):
    h1 = dropout(rectify(b1+T.dot(X, s*w_h1)),p)
    pyx = 1.0 / (1.0 + T.exp(-T.dot(h1, w_o) - bo))
    return pyx

def SVMclassify(trX, teX, trY, teY):
    clf = svm.SVC(C=1.9, gamma=0.01)   
    #clf.fit(trX, np.argmax(trY, axis=1))
    clf.fit(trX, trY)
    teY1 = clf.predict(teX)    
    #print np.mean(np.argmax(teY, axis=1) == teY1)
    print np.mean(teY == teY1)
    return teY1

def ComputeComplexity(n_layers):
    comp = 0
    for i in range(len(n_layers)-1):
        comp += n_layers[i]*n_layers[i+1]
    return(comp)
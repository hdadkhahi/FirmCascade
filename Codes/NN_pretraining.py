
from __future__ import division
import cascade_functions as CF
import time
import theano
from theano import tensor as T
import numpy as np
import lasagne


def NN_pretraining(trX2,teP,teX1,teY1,K1,K2):
    
    K1_vector = [K1]
    
    n_it = 10000
    time1 = np.zeros((len(K1_vector),1))
    accuracy1 = np.zeros((len(K1_vector),1))
    
    inds_true = np.where( teY1 == 1 )[0]    
    
    for i,K1 in enumerate(K1_vector):

            (N,D) = trX2.shape
            X = T.fmatrix()
            Y = T.fvector()
                        
            w_h1 = CF.init_weights((D,  K1))
            b1   = CF.init_weights((K1,  ))
            w_h2 = CF.init_weights((K1, K2))  
            b2   = CF.init_weights((K2,  ))
            w_o  = CF.init_weights((K2, ))
            bo = theano.shared(CF.floatX(np.random.randn(1) * 0.01), broadcastable=(True,))
            
            pygx = CF.model(X, w_h1, w_h2, w_o, b1,b2,bo,0,1)
            # yhat_second = T.argmax(pygx, axis=1)
            yhat_second = (pygx > 0.5)
            
            # cost_second = T.mean(T.nnet.categorical_crossentropy(pygx, Y))
            cost_second = T.mean(T.nnet.binary_crossentropy(pygx, Y))
            params_second = [w_h1,w_h2, w_o,b1,b2,bo]

            updates_second = lasagne.updates.rmsprop(cost_second, params_second, learning_rate=0.002, rho=0.9, epsilon=1e-06)
            
            train_second = theano.function(inputs=[X, Y], outputs=cost_second, updates=updates_second, allow_input_downcast=True)
            predict_second = theano.function(inputs=[X], outputs=yhat_second, allow_input_downcast=True)
            
            sgd_iters = 1500
            for j in range(sgd_iters):
                c = train_second(trX2, teP)
                print(c)
            
            teY3 = predict_second(teX1)
            
            start1 = time.clock()
            for t in range(n_it):
                teY3 = predict_second(teX1)
            end1 = time.clock()
            time1[i] = (end1 - start1)
            
            accuracy1[i] = np.mean(teY1 == teY3)
            inds_second = np.where( teY3 == 1 )[0]
            int_result2 = np.intersect1d(inds_second,inds_true)
            print("nzs:%d,true nzs:%d,intersection:%d" %(inds_second.shape[0],inds_true.shape[0],int_result2.shape[0]))
            r2 = int_result2.shape[0] / inds_true.shape[0]
            p2 = int_result2.shape[0] / inds_second.shape[0]
            print("recall = %f, precision = %f, accuracy = %f" %(r2,p2,accuracy1))
            F1 = 2*r2*p2/(r2 + p2)
            
    return w_h1, w_h2, w_o, b1, b2, bo, time1, accuracy1, F1
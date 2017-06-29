
from __future__ import division
import cascade_functions as CF
import time
import theano
from theano import tensor as T
import numpy as np
import lasagne


def second_stage_pretraining(trX1, trY1, teX1, teY1, trX11, teX11, K1, w_h1, w_h2, w_o, b1, b2, bo, plambda, a):
    
    lambda_vector = plambda
    
    n_it = 10000
    time1 = np.zeros((len(lambda_vector),1))
    accuracy1 = np.zeros((len(lambda_vector),1))
    F1 = np.zeros((len(lambda_vector),1))
    nnz = np.zeros((len(lambda_vector),1))
    
    for i, plambda in enumerate(lambda_vector):
                
        (N,D) = trX11.shape
                    
        X = T.fmatrix()
        F = T.fmatrix()
        Y = T.fvector()
        
        v_h1 = CF.init_weights((D,  K1))
        c1   = CF.init_weights((K1,  ))
        v_o  = CF.init_weights((K1, ))
        co = theano.shared(CF.floatX(np.random.randn(1) * 0.01), broadcastable=(True,))
                
        pygx1 = CF.model3(F, v_h1, v_o, c1, co, 0, 1)
        pygx = CF.model(X, w_h1, w_h2, w_o, b1, b2, bo, 0, 1)
        
        yhat1 = (pygx1 > 0.5)
        yhat = (pygx > 0.5)
        
        f = lambda x, a: 1/(1+T.exp(-a*(x-0.5)))
        
        pygx_final = (1-f(pygx1,a))*pygx1 + f(pygx1,a)*pygx
        reg = T.mean(f(pygx1,a))  
        cost = T.mean(T.nnet.binary_crossentropy(pygx_final, Y)) + plambda*reg 

        params = [v_h1, v_o, c1, co]
        # updates = lasagne.updates.rmsprop(cost, params, learning_rate=0.01/5, rho=0.9, epsilon=1e-06)
        updates = lasagne.updates.adagrad(cost, params, learning_rate=1, epsilon=1e-06)
        
        # train = theano.function(inputs=[F, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        train = theano.function(inputs=[X, F, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        reg_value = theano.function(inputs=[F], outputs=reg, allow_input_downcast=True)
        
        predict_first = theano.function(inputs=[F], outputs=yhat1, allow_input_downcast=True)
        predict_second = theano.function(inputs=[X], outputs=yhat, allow_input_downcast=True)
        
        max_iter = 1000
        for j in range(max_iter):
            # c = train(trX1, trY1)
            c = train(trX1, trX11, trY1) 
            # c = train(trX11, trY1)
            # r = reg_value(trX1)
            r = reg_value(trX11) 
            print(c-plambda*r,plambda*r)
            # cost = train(trX1, trY1)
        
        start1 = time.clock()
        for t in range(n_it):
            teQ1 = predict_first(teX11)
        end1 = time.clock()
        time1[i] = end1 - start1
        inds_test = np.where(teQ1 == 1)[0]
        nnz[i] = inds_test.shape[0]

        # check that we get 100 percent recall from the first stage
        inds_true = np.where( teY1 == 1 )[0]
        int_result = np.intersect1d(inds_test,inds_true)
        print("first stage nzs:%d,true nzs:%d,intersection:%d" %(inds_test.shape[0],inds_true.shape[0],int_result.shape[0]))
        r1 = int_result.shape[0] / inds_true.shape[0]
        p1 = int_result.shape[0] / inds_test.shape[0]
        a1 = np.mean(teY1 == teQ1)
        print("first stage: recall = %f, precision = %f, accuracy = %f" %(r1,p1,a1))
        
        teX2 = teX1[inds_test,:]
        
        start1 = time.clock()
        for t in range(n_it):
            teQ2 = predict_second(teX2)    
        end1 = time.clock()
        time1[i] += end1 - start1
            
        teY3 = np.zeros(teY1.shape,dtype = int)
        teY3.fill(0)
        teY3[inds_test] = teQ2
        accuracy1[i] = np.mean(teY1 == teY3)    
        
        inds_second = np.where( teY3 == 1 )[0]
        int_result2 = np.intersect1d(inds_second,inds_true)
        print("second stage nzs:%d,true nzs:%d,intersection:%d" %(inds_second.shape[0],inds_true.shape[0],int_result2.shape[0]))
        r2 = int_result2.shape[0] / inds_true.shape[0]
        p2 = int_result2.shape[0] / inds_second.shape[0]
        print("second stage: recall = %f, precision = %f, accuracy = %f" %(r2,p2,accuracy1[i]))
        F1[i] = 2*r2*p2/(r2 + p2)
        
    return v_h1, v_o, c1, co, time1, accuracy1, F1    
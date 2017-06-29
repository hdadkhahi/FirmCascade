
from __future__ import division
import cascade_functions as CF
import time
import theano
from theano import tensor as T
import numpy as np
import lasagne


def ComputeComplexity(n_layers):
    comp = 0
    for i in range(len(n_layers)-1):
        comp += n_layers[i]*n_layers[i+1]
    return(comp)


def soft_cascade_LR_1LNN(trX1, trY1, teX1, teY1, trX2, teX2, lambda_vector, K1):

    (N, D1) = trX2.shape
    D = trX1.shape[1]
    C = 2
    t1 = ComputeComplexity([D1, C])
    t2 = ComputeComplexity([D, K1, C])

    n_it = 10000
    time1 = np.zeros((len(lambda_vector),1))
    accuracy1 = np.zeros((len(lambda_vector),1))
    F1 = np.zeros((len(lambda_vector),1))
    nnz_first = np.zeros((len(lambda_vector),1))

    for i, plambda in enumerate(lambda_vector):

        X = T.fmatrix()
        F = T.fmatrix()
        Y = T.fvector()

        w_l = CF.init_weights((D1,))
        b_l = theano.shared(CF.floatX(np.random.randn(1) * 0.01), broadcastable=(True,))
        # w_l.set_value(np.zeros((D1,)))
        # b_l.set_value(np.zeros((1,)))

        w_h1 = CF.init_weights((D, K1))
        b1 = CF.init_weights((K1,))
        w_o = CF.init_weights((K1,))
        bo = theano.shared(CF.floatX(np.random.randn(1) * 0.01), broadcastable=(True,))

        pygx1 = CF.model00(F, w_l, b_l)
        pygx2 = CF.model3(X, w_h1, w_o, b1, bo, 0, 1)
        pygx_final = pygx1 * pygx2

        yhat1 = (pygx1 > 0.5)
        yhat = (pygx2 > 0.5)

        reg = T.mean(t1 + t2*pygx1)
        cost = T.mean(T.nnet.binary_crossentropy(pygx_final, Y)) + plambda*reg

        params = [w_l, b_l, w_h1, w_o, b1, bo]
        updates = lasagne.updates.rmsprop(cost, params, learning_rate=0.001*5, rho=0.9, epsilon=1e-06)
        # updates = lasagne.updates.adagrad(cost, params, learning_rate=1, epsilon=1e-06)

        train = theano.function(inputs=[X, F, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        reg_value = theano.function(inputs=[F], outputs=reg, allow_input_downcast=True)

        predict_first = theano.function(inputs=[F], outputs=yhat1, allow_input_downcast=True)
        predict_second = theano.function(inputs=[X], outputs=yhat, allow_input_downcast=True)

        max_iter = 300
        for j in range(max_iter):
            c = train(trX1, trX2, trY1)
            r = reg_value(trX2)
            print(c-plambda*r,plambda*r)

        start1 = time.clock()
        for t in range(n_it):
            teQ1 = predict_first(teX2)
        end1 = time.clock()
        time1[i] = end1 - start1
        inds_test = np.where(teQ1 == 1)[0]
        nnz_first[i] = inds_test.shape[0]

        # check that we get 100 percent recall from the first stage
        inds_true = np.where( teY1 == 1 )[0]
        int_result = np.intersect1d(inds_test,inds_true)
        print("first stage nzs:%d,true nzs:%d,intersection:%d" %(inds_test.shape[0],inds_true.shape[0],int_result.shape[0]))
        r1 = int_result.shape[0] / inds_true.shape[0]
        p1 = int_result.shape[0] / inds_test.shape[0]
        a1 = np.mean(teY1 == teQ1)
        print("first stage: recall = %f, precision = %f, accuracy = %f" %(r1,p1,a1))

        teX11 = teX1[inds_test,:]

        start1 = time.clock()
        for t in range(n_it):
            teQ2 = predict_second(teX11)
        end1 = time.clock()
        time1[i] += end1 - start1

        teY2 = np.zeros(teY1.shape,dtype = int)
        teY2.fill(0)
        teY2[inds_test] = teQ2

        inds_second = np.where( teY2 == 1 )[0]
        int_result = np.intersect1d(inds_second, inds_true)
        print("second stage nzs:%d,true nzs:%d,intersection:%d" %(inds_second.shape[0],inds_true.shape[0],int_result.shape[0]))
        r2 = int_result.shape[0] / inds_true.shape[0]
        p2 = int_result.shape[0] / inds_second.shape[0]
        a2 = np.mean(teY1 == teY2)
        print("second stage: recall = %f, precision = %f, accuracy = %f" %(r2,p2,a2))
        F1[i] = 2 * r2 * p2 / (r2 + p2)
        accuracy1[i] = a2

    return time1, accuracy1, F1, nnz_first


from __future__ import division
import cascade_functions as CF
import time
import theano
from theano import tensor as T
import numpy as np
import lasagne
from sklearn.metrics import roc_auc_score


def ComputeComplexity(n_layers):
    comp = 0
    for i in range(len(n_layers)-1):
        comp += n_layers[i]*n_layers[i+1]
    return(comp)


def soft_cascade_rw(trX, trY, teX, teY, trX1, teX1, trX2, teX2, beta, K):

    lambda_vector = beta
    C = 2

    (N, D1) = trX1.shape
    (N, D2) = trX2.shape
    (N, D) = trX.shape

    t1 = ComputeComplexity([D1, C])
    t2 = ComputeComplexity([D2, C])
    t3 = ComputeComplexity([D, K, C])

    n_it = 10000
    time1 = np.zeros((len(lambda_vector),1))
    accuracy1 = np.zeros((len(lambda_vector),1))
    F1 = np.zeros((len(lambda_vector),1))
    nnz = np.zeros((len(lambda_vector),1))

    for i, plambda in enumerate(lambda_vector):

        X = T.fmatrix()
        F = T.fmatrix()
        E = T.fmatrix()
        Y = T.fvector()

        w_l = CF.init_weights((D1,))
        b_l = theano.shared(CF.floatX(np.random.randn(1) * 0.01), broadcastable=(True,))
        w_l.set_value(np.zeros((D1,)))
        b_l.set_value(np.zeros((1,)))

        v_l = CF.init_weights((D2,))
        c_l = theano.shared(CF.floatX(np.random.randn(1) * 0.01), broadcastable=(True,))
        v_l.set_value(np.zeros((D2,)))
        c_l.set_value(np.zeros((1,)))

        w_h1 = CF.init_weights((D, K))
        b1 = CF.init_weights((K,))
        w_o = CF.init_weights((K,))
        bo = theano.shared(CF.floatX(np.random.randn(1) * 0.01), broadcastable=(True,))

        pygx1 = CF.model00(F, w_l, b_l)
        pygx2 = CF.model00(E, v_l, c_l)
        pygx = CF.model3(X, w_h1, w_o, b1, bo, 0, 1)

        yhat1 = (pygx1 > 0.5)
        yhat2 = (pygx2 > 0.5)
        yhat = (pygx > 0.5)

        f = lambda x, a: 1 / (1 + T.exp(-a * (x - 0.5)))

        pygx_final = pygx1 * pygx2 * pygx

        reg = T.mean( t1 + t2*pygx1 + t3*pygx1*pygx2 )
        cost = T.mean(T.nnet.binary_crossentropy(pygx_final, Y)) + plambda*reg

        params = [w_l, b_l, v_l, c_l, w_h1, w_o, b1, bo]

        # updates = lasagne.updates.adagrad(cost, params, learning_rate=1/5, epsilon=1e-06)
        updates = lasagne.updates.rmsprop(cost, params, learning_rate=0.075, rho=0.9, epsilon=1e-06)

        train = theano.function(inputs=[X, F, E, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        reg_value = theano.function(inputs=[F, E], outputs=reg, allow_input_downcast=True)

        predict_first = theano.function(inputs=[F], outputs=yhat1, allow_input_downcast=True)
        predict_second = theano.function(inputs=[E], outputs=yhat2, allow_input_downcast=True)
        predict_final = theano.function(inputs=[X], outputs=yhat, allow_input_downcast=True)
        predict_prob = theano.function(inputs=[X, F, E], outputs=pygx_final, allow_input_downcast=True)

        max_iter = 400
        for j in range(max_iter):
            c = train(trX, trX1, trX2, trY)
            r = reg_value(trX1, trX2)
            print(c-plambda*r, plambda*r)

        probs = predict_prob(teX, teX1, teX2)
        AUC = roc_auc_score(teY, probs)

        start1 = time.clock()
        for t in range(n_it):
            teQ1 = predict_first(teX1)
        end1 = time.clock()
        time1[i] = end1 - start1
        inds_test1 = np.where(teQ1 == 1)[0]
        nnz[i] = inds_test1.shape[0]

        inds_true = np.where( teY == 1 )[0]
        int_result1 = np.intersect1d(inds_test1,inds_true)
        print("first stage nzs:%d,true nzs:%d,intersection:%d" %(inds_test1.shape[0],inds_true.shape[0],int_result1.shape[0]))

        teX2 = teX2[inds_test1, :]

        start1 = time.clock()
        for t in range(n_it):
            teQ2 = predict_second(teX2)
        end1 = time.clock()
        time1[i] = end1 - start1
        inds_test2 = np.where(teQ2 == 1)[0]
        nnz[i] = inds_test2.shape[0]

        int_result2 = np.intersect1d(inds_test1[inds_test2], inds_true)
        print("second stage nzs:%d,true nzs:%d,intersection:%d" % (
        inds_test2.shape[0], inds_true.shape[0], int_result2.shape[0]))


        teX = teX[inds_test1[inds_test2],:]

        start1 = time.clock()
        for t in range(n_it):
            teP = predict_final(teX)
        end1 = time.clock()
        time1[i] += end1 - start1

        teY3 = np.zeros(teY.shape, dtype=int)
        teY3.fill(0)
        teY3[inds_test1[inds_test2]] = teP
        accuracy1[i] = np.mean(teY == teY3)

        inds_second = np.where( teY3 == 1 )[0]
        int_result = np.intersect1d(inds_second,inds_true)
        print("final stage nzs:%d,true nzs:%d,intersection:%d" %(inds_second.shape[0],inds_true.shape[0],int_result.shape[0]))
        r = int_result.shape[0] / inds_true.shape[0]
        p = int_result.shape[0] / inds_second.shape[0]
        print("final stage: recall = %f, precision = %f, accuracy = %f" %(r,p,accuracy1[i]))
        F1[i] = 2*r*p/(r + p)

    return time1, accuracy1, F1, nnz, AUC
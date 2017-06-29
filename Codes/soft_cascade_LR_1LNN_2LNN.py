
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


def soft_cascade_LR_1LNN_2LNN(trX1, trY1, teX1, teY1, trX2, teX2, trX3, teX3, lambda_vector, K, K1, K2):

    (N, D1) = trX3.shape
    D = trX1.shape[1]
    C = 2
    t1 = ComputeComplexity([D1, C])
    t2 = ComputeComplexity([D, K1, C])
    t3 = ComputeComplexity([D, K1, K2, C])

    n_it = 10000
    time1 = np.zeros((len(lambda_vector),1))
    accuracy1 = np.zeros((len(lambda_vector),1))
    F1 = np.zeros((len(lambda_vector),1))
    nnz_first = np.zeros((len(lambda_vector),1))
    nnz_second = np.zeros((len(lambda_vector), 1))

    for i, plambda in enumerate(lambda_vector):

        X = T.fmatrix()
        F = T.fmatrix()
        E = T.fmatrix()
        Y = T.fvector()

        w_l = CF.init_weights((D1,))
        b_l = theano.shared(CF.floatX(np.random.randn(1) * 0.01), broadcastable=(True,))

        w_h1 = CF.init_weights((D, K))
        b1 = CF.init_weights((K,))
        w_o = CF.init_weights((K,))
        bo = theano.shared(CF.floatX(np.random.randn(1) * 0.01), broadcastable=(True,))

        v_h1 = CF.init_weights((D, K1))
        c1 = CF.init_weights((K1,))
        v_h2 = CF.init_weights((K1, K2))
        c2 = CF.init_weights((K2,))
        v_o = CF.init_weights((K2,))
        co = theano.shared(CF.floatX(np.random.randn(1) * 0.01), broadcastable=(True,))

        pygx1 = CF.model00(F, w_l, b_l)
        pygx2 = CF.model3(X, w_h1, w_o, b1, bo, 0, 1)
        pygx3 = CF.model(E, v_h1, v_h2, v_o, c1, c2, co, 0, 1)
        pygx_final = pygx1 * pygx2 * pygx3

        yhat1 = (pygx1 > 0.5)
        yhat2 = (pygx2 > 0.5)
        yhat3 = (pygx3 > 0.5)

        reg = T.mean(t1 + t2*pygx1 + t3*pygx1*pygx2)
        cost = T.mean(T.nnet.binary_crossentropy(pygx_final, Y)) + plambda*reg

        params = [w_l, b_l, w_h1, w_o, b1, bo, v_h1, v_h2, v_o, c1, c2, co]
        # updates = lasagne.updates.rmsprop(cost, params, learning_rate=0.01/2, rho=0.9, epsilon=1e-06)
        updates = lasagne.updates.adagrad(cost, params, learning_rate=1/30, epsilon=1e-06)

        train = theano.function(inputs=[E, X, F, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        reg_value = theano.function(inputs=[X, F], outputs=reg, allow_input_downcast=True)

        predict_first = theano.function(inputs=[F], outputs=yhat1, allow_input_downcast=True)
        predict_second = theano.function(inputs=[X], outputs=yhat2, allow_input_downcast=True)
        predict_third = theano.function(inputs=[E], outputs=yhat3, allow_input_downcast=True)

        max_iter = 3000
        for j in range(max_iter):
            c = train(trX1, trX2, trX3, trY1)
            r = reg_value(trX2, trX3)
            print(c-plambda*r, plambda*r)

        start1 = time.clock()
        for t in range(n_it):
            teQ1 = predict_first(teX3)
        end1 = time.clock()
        time1[i] = end1 - start1
        inds_test = np.where(teQ1 == 1)[0]
        nnz_first[i] = inds_test.shape[0]

        # check that we get 100 percent recall from the first stage
        inds_true = np.where(teY1 == 1)[0]
        int_result = np.intersect1d(inds_test, inds_true)
        print("first stage nzs:%d,true nzs:%d,intersection:%d" % (inds_test.shape[0], inds_true.shape[0], int_result.shape[0]))
        r1 = int_result.shape[0] / inds_true.shape[0]
        p1 = int_result.shape[0] / inds_test.shape[0]
        a1 = np.mean(teY1 == teQ1)
        print("first stage: recall = %f, precision = %f, accuracy = %f" % (r1, p1, a1))

        teX22 = teX2[inds_test, :]

        start1 = time.clock()
        for t in range(n_it):
            teQ2 = predict_second(teX22)
        end1 = time.clock()
        time1[i] += end1 - start1
        inds_test2 = np.where(teQ2 == 1)[0]
        nnz_second[i] = inds_test2.shape[0]

        teY2 = np.zeros(teY1.shape, dtype=int)
        teY2.fill(0)
        teY2[inds_test] = teQ2

        inds_second = np.where(teY2 == 1)[0]
        int_result = np.intersect1d(inds_second, inds_true)
        print("second stage nzs:%d,true nzs:%d,intersection:%d" % (
        inds_second.shape[0], inds_true.shape[0], int_result.shape[0]))
        r2 = int_result.shape[0] / inds_true.shape[0]
        p2 = int_result.shape[0] / inds_second.shape[0]
        a2 = np.mean(teY1 == teY2)
        print("second stage: recall = %f, precision = %f, accuracy = %f" % (r2, p2, a2))

        # teX1 = teX1[inds_test2,:]
        teX11 = teX1[inds_test[inds_test2], :]

        start1 = time.clock()
        for t in range(n_it):
            teQ3 = predict_third(teX11)
        end1 = time.clock()
        time1[i] += end1 - start1

        teY3 = np.zeros(teY1.shape, dtype=int)
        teY3.fill(0)
        teY3[inds_test[inds_test2]] = teQ3
        accuracy1[i] = np.mean(teY1 == teY3)

        inds_third = np.where(teY3 == 1)[0]
        int_result2 = np.intersect1d(inds_third, inds_true)
        print("third stage nzs:%d,true nzs:%d,intersection:%d" % (
        inds_third.shape[0], inds_true.shape[0], int_result2.shape[0]))
        r3 = int_result2.shape[0] / inds_true.shape[0]
        p3 = int_result2.shape[0] / inds_third.shape[0]
        print("third stage: recall = %f, precision = %f, accuracy = %f" % (r3, p3, accuracy1[i]))
        F1[i] = 2 * r3 * p3 / (r3 + p3)

    return time1, accuracy1, F1, nnz_first, nnz_second

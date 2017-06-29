
from __future__ import division
import cascade_functions as CF
import time
import theano
from theano import tensor as T
import numpy as np
import lasagne
from sklearn.metrics import roc_auc_score


def tree_soft_cascade_v1(trX, trY, teX, teY, trX1, teX1, trX2, teX2, plambda, K, K1, K2):

    lambda_vector = plambda

    n_it = 10000
    time1 = np.zeros((len(lambda_vector),1))
    accuracy1 = np.zeros((len(lambda_vector),1))
    F1 = np.zeros((len(lambda_vector),1))
    nnz = np.zeros((len(lambda_vector),1))

    for i, plambda in enumerate(lambda_vector):

        (N, D1) = trX1.shape
        (N, D2) = trX2.shape
        (N, D) = trX.shape

        X = T.fmatrix()
        V = T.fmatrix()
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

        v_h1 = CF.init_weights((D, K))
        c1 = CF.init_weights((K,))
        v_o = CF.init_weights((K,))
        co = theano.shared(CF.floatX(np.random.randn(1) * 0.01), broadcastable=(True,))

        w_h1 = CF.init_weights((D, K1))
        b1 = CF.init_weights((K1,))
        w_h2 = CF.init_weights((K1, K2))
        b2 = CF.init_weights((K2,))
        w_o = CF.init_weights((K2,))
        bo = theano.shared(CF.floatX(np.random.randn(1) * 0.01), broadcastable=(True,))

        pygx1 = CF.model00(F, w_l, b_l)
        pygx2 = CF.model00(E, v_l, c_l)
        pygx3 = CF.model3(V, v_h1, v_o, c1, co, 0, 1)
        pygx = CF.model(X, w_h1, w_h2, w_o, b1, b2, bo, 0, 1)

        yhat1 = (pygx1 > 0.5)
        yhat2 = (pygx2 > 0.5)
        yhat3 = (pygx3 > 0.5)
        yhat = (pygx > 0.5)

        pygx_final = pygx1 * pygx2 * pygx3 * pygx

        kappa1 = 1/20
        kappa2 = 1/20
        kappa3 = 1/10
        kappa5 = 1

        reg = T.mean(kappa1 + kappa2 + kappa3 * pygx1 * pygx2 + kappa5 * pygx1 * pygx2 * pygx3)
        cost = T.mean(T.nnet.binary_crossentropy(pygx_final, Y)) + plambda*reg

        params = [w_l, b_l, v_l, c_l, v_h1, v_o, c1, co, w_h1, w_h2, w_o, b1, b2, bo]

        # updates = lasagne.updates.adagrad(cost, params, learning_rate=1, epsilon=1e-06)
        updates = lasagne.updates.rmsprop(cost, params, learning_rate=0.1/5, rho=0.9, epsilon=1e-06)

        train = theano.function(inputs=[X, V, F, E, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        reg_value = theano.function(inputs=[V, F, E], outputs=reg, allow_input_downcast=True)

        predict_first = theano.function(inputs=[F], outputs=yhat1, allow_input_downcast=True)
        predict_second = theano.function(inputs=[E], outputs=yhat2, allow_input_downcast=True)
        predict_third = theano.function(inputs=[V], outputs=yhat3, allow_input_downcast=True)
        predict_final = theano.function(inputs=[X], outputs=yhat, allow_input_downcast=True)

        max_iter = 500
        for j in range(max_iter):
            c = train(trX, trX, trX1, trX2, trY)
            r = reg_value(trX, trX1, trX2)
            print(c-plambda*r, plambda*r)

        start1 = time.clock()
        for t in range(n_it):
            teQ1 = predict_first(teX1)
        end1 = time.clock()
        time1[i] += end1 - start1
        inds_test1 = np.where(teQ1 == 1)[0]
        nnz[i] = inds_test1.shape[0]

        inds_true = np.where(teY == 1)[0]
        int_result1 = np.intersect1d(inds_test1, inds_true)
        print("first stage nzs:%d,true nzs:%d,intersection:%d" %(inds_test1.shape[0],inds_true.shape[0],int_result1.shape[0]))
        r1 = int_result1.shape[0] / inds_true.shape[0]
        p1 = int_result1.shape[0] / inds_test1.shape[0]
        a1 = np.mean(teY == teQ1)
        print("first stage: recall = %f, precision = %f, accuracy = %f" %(r1,p1,a1))

        start1 = time.clock()
        for t in range(n_it):
            teQ2 = predict_second(teX2)
        end1 = time.clock()
        time1[i] += end1 - start1
        inds_test2 = np.where(teQ2 == 1)[0]
        nnz[i] = inds_test2.shape[0]

        int_result2 = np.intersect1d(inds_test2, inds_true)
        print("second stage nzs:%d,true nzs:%d,intersection:%d" % (
        inds_test2.shape[0], inds_true.shape[0], int_result2.shape[0]))
        r2 = int_result2.shape[0] / inds_true.shape[0]
        p2 = int_result2.shape[0] / inds_test2.shape[0]
        a2 = np.mean(teY == teQ2)
        print("second stage: recall = %f, precision = %f, accuracy = %f" % (r2, p2, a2))

        inds_test = np.intersect1d(inds_test1, inds_test2)
        tps = np.intersect1d(inds_test, inds_true)
        print("intersects of first-second branches = %d, true positives = %d" % (inds_test.shape[0], tps.shape[0]))

        teXX = teX[inds_test,:]

        start1 = time.clock()
        for t in range(n_it):
            teQ3 = predict_third(teXX)
        end1 = time.clock()
        time1[i] += end1 - start1
        inds_test3 = np.where(teQ3 == 1)[0]
        nnz[i] = inds_test3.shape[0]

        int_result33 = np.intersect1d(inds_test[inds_test3], inds_true)
        print("third stage nzs:%d,true nzs:%d,intersection:%d" % (
            inds_test3.shape[0], inds_true.shape[0], int_result33.shape[0]))

        teXXX = teXX[inds_test3, :]

        start1 = time.clock()
        for t in range(n_it):
            teP = predict_final(teXXX)
        end1 = time.clock()
        time1[i] += end1 - start1

        teY3 = np.zeros(teY.shape, dtype=int)
        teY3.fill(0)
        teY3[inds_test[inds_test3]] = teP
        accuracy1[i] = np.mean(teY == teY3)

        inds_final = np.where(teY3 == 1)[0]
        int_result = np.intersect1d(inds_final, inds_true)
        print("final stage nzs:%d,true nzs:%d,intersection:%d" %(inds_final.shape[0],inds_true.shape[0],int_result.shape[0]))
        r = int_result.shape[0] / inds_true.shape[0]
        p = int_result.shape[0] / inds_final.shape[0]
        print("final stage: recall = %f, precision = %f, accuracy = %f" %(r,p,accuracy1[i]))
        F1[i] = 2*r*p/(r + p)

    return time1, accuracy1, F1, nnz
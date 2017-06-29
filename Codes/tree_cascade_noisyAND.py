
from __future__ import division
import cascade_functions as CF
import time
import theano
from theano import tensor as T
import numpy as np
import lasagne
from sklearn.metrics import roc_auc_score


def tree_cascade_noisyAND(trX, trY, teX, teY, trX1, teX1, trX2, teX2, w_h1, w_o, b1, bo, plambda, a):

    lambda_vector = plambda

    n_it = 10000
    time1 = np.zeros((len(lambda_vector),1))
    accuracy1 = np.zeros((len(lambda_vector),1))
    F1 = np.zeros((len(lambda_vector),1))
    nnz = np.zeros((len(lambda_vector),1))

    for i, plambda in enumerate(lambda_vector):

        (N, D1) = trX1.shape
        (N, D2) = trX2.shape

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

        pygx1 = CF.model00(F, w_l, b_l)
        pygx2 = CF.model00(E, v_l, c_l)
        pygx = CF.model3(X, w_h1, w_o, b1, bo,0,1)

        yhat1 = (pygx1 > 0.5)
        yhat2 = (pygx2 > 0.5)
        yhat = (pygx > 0.5)

        f = lambda x, a: 1 / (1 + T.exp(-a * (x - 0.5)))

        pygx_final = (1 - f(pygx1, a)*f(pygx2, a)) * pygx1*pygx2 + f(pygx1, a)*f(pygx2, a) * pygx

        reg = T.mean( f(pygx1,a)*f(pygx2,a) )
        cost = T.mean(T.nnet.binary_crossentropy(pygx_final, Y)) + plambda*reg

        # params = [w_l, b_l, v_l, c_l]
        params = [w_l, b_l, v_l, c_l, w_h1, w_o, b1, bo]
        # params = [w_h1, w_h2, w_o, w_l, b1, b2, bo, b_l]
        # params = [w_h1, w_h2, w_o, w_l, b1, b2, bo]

        # updates = lasagne.updates.adagrad(cost, params, learning_rate=1/5, epsilon=1e-06)
        updates = lasagne.updates.rmsprop(cost, params, learning_rate=0.1/2, rho=0.9, epsilon=1e-06)

        train = theano.function(inputs=[X, F, E, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        reg_value = theano.function(inputs=[F, E], outputs=reg, allow_input_downcast=True)

        predict_first = theano.function(inputs=[F], outputs=yhat1, allow_input_downcast=True)
        predict_second = theano.function(inputs=[E], outputs=yhat2, allow_input_downcast=True)
        predict_final = theano.function(inputs=[X], outputs=yhat, allow_input_downcast=True)
        predict_prob = theano.function(inputs=[X, F, E], outputs=pygx_final, allow_input_downcast=True)

        max_iter = 300
        for j in range(max_iter):
            c = train(trX, trX1, trX2, trY)
            r = reg_value(trX1, trX2)
            print(c-plambda*r, plambda*r)

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
        r1 = int_result1.shape[0] / inds_true.shape[0]
        p1 = int_result1.shape[0] / inds_test1.shape[0]
        a1 = np.mean(teY == teQ1)
        print("first stage: recall = %f, precision = %f, accuracy = %f" %(r1,p1,a1))

        start1 = time.clock()
        for t in range(n_it):
            teQ2 = predict_second(teX2)
        end1 = time.clock()
        time1[i] = end1 - start1
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
        print("intersects of first-second stages = %d, true positives = %d" % (inds_test.shape[0], tps.shape[0]))

        probs = predict_prob(teX, teX1, teX2)
        AUC = roc_auc_score(teY,probs)

        teX = teX[inds_test,:]

        start1 = time.clock()
        for t in range(n_it):
            teP = predict_final(teX)
        end1 = time.clock()
        time1[i] += end1 - start1

        teY3 = np.zeros(teY.shape, dtype=int)
        teY3.fill(0)
        teY3[inds_test] = teP
        accuracy1[i] = np.mean(teY == teY3)

        inds_second = np.where( teY3 == 1 )[0]
        int_result = np.intersect1d(inds_second,inds_true)
        print("final stage nzs:%d,true nzs:%d,intersection:%d" %(inds_second.shape[0],inds_true.shape[0],int_result.shape[0]))
        r = int_result.shape[0] / inds_true.shape[0]
        p = int_result.shape[0] / inds_second.shape[0]
        print("final stage: recall = %f, precision = %f, accuracy = %f" %(r,p,accuracy1[i]))
        F1[i] = 2*r*p/(r + p)

    return AUC, time1, accuracy1, F1, nnz
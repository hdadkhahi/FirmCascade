
from __future__ import division
import cascade_functions as CF
import time
import theano
from theano import tensor as T
import numpy as np
import lasagne


# z-normalization of data
def z_normalize(trX):
    m = np.mean(trX, axis = 0)
    v = np.std(trX, axis = 0)
    trX = (trX - m) / v
    return trX


# trX1, trY1, teX1, teY1: data used in the second stage. trX2, teX2: data used in the first stage.
def cascade_two_stage(trX1, trY1, teX1, teY1, trX2, teX2, w_h1, w_o, b1, bo, plambda, a):

    lambda_vector = plambda

    # number of iterations for prediction:
    n_it = 10000
    # prediction time:
    time1 = np.zeros((len(lambda_vector),1))
    # accuracy:
    accuracy1 = np.zeros((len(lambda_vector),1))
    # F1 score:
    F1 = np.zeros((len(lambda_vector),1))
    # number of non-zeros sent to the second stage:
    nnz = np.zeros((len(lambda_vector),1))

    for i, plambda in enumerate(lambda_vector):

        # N: number of training data points, D: number of dimensions/features in first stage data
        (N, D) = trX2.shape

        # second stage training data:
        X = T.fmatrix()
        # first stage training data:
        F = T.fmatrix()
        # lables for training data:
        Y = T.fvector()

        # random initialization of LR paramters:
        w_l = CF.init_weights((D,))
        b_l = theano.shared(CF.floatX(np.random.randn(1) * 0.01), broadcastable=(True,))
        # zero initialization of LR parameters:
        w_l.set_value(np.zeros((D,)))
        b_l.set_value(np.zeros((1,)))

        # define LR model:
        pygx1 = CF.model00(F, w_l, b_l)
        # define 2LNN model:
        # pygx = CF.model(X, w_h1, w_h2, w_o, b1, b2, bo, 0, 1)
        # define 1LNN model:
        pygx = CF.model3(X, w_h1, w_o, b1, bo, 0, 1)

        # hard threshold cascade: thresholding of output probabilities
        yhat1 = (pygx1 > 0.5) # output of first stage
        yhat = (pygx > 0.5) # output of second stage

        # definition of the gating function:
        f = lambda x, a: 1 / (1 + T.exp(-a * (x - 0.5)))

        # output probability of the cascade:
        pygx_final = (1-f(pygx1,a))*pygx1 + f(pygx1,a)*pygx

        # regularization term:
        reg = T.mean(f(pygx1,a))
        # objective function:
        cost = T.mean(T.nnet.binary_crossentropy(pygx_final, Y)) + plambda*reg

        # parameters of the optimization problem:
        params = [w_l, b_l]
        # params = [w_h1, w_o, w_l, b1, bo, b_l]
        # params = [w_h1, w_h2, w_o, w_l, b1, b2, bo, b_l]
        # params = [w_h1, w_h2, w_o, w_l, b1, b2, bo]

        # updates = lasagne.updates.rmsprop(cost, params, learning_rate=0.0004, rho=0.9, epsilon=1e-06)
        updates = lasagne.updates.adagrad(cost, params, learning_rate=1, epsilon=1e-06)

        # theano function for training:
        train = theano.function(inputs=[X, F, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        reg_value = theano.function(inputs=[F], outputs=reg, allow_input_downcast=True)

        # theano function for prediction in first stage:
        predict_first = theano.function(inputs=[F], outputs=yhat1, allow_input_downcast=True)
        # theano function for prediction in second stage:
        predict_second = theano.function(inputs=[X], outputs=yhat, allow_input_downcast=True)

        # number of steps in SGD:
        max_iter = 5000

        # iterations for SGD/training:
        for j in range(max_iter):
            c = train(trX1, trX2, trY1)
            r = reg_value(trX2)
            print(c, c-plambda*r, plambda*r)
            # cost = train(trX1, trY1)

        # prediction for first stage:
        start1 = time.clock()
        for t in range(n_it):
            teQ1 = predict_first(teX2)
            # teQ1 = teX1.dot(w_l.get_value()) + b_check >= 0
            # teQ1 = np.dot(teX2,w_l.get_value()) + b_l.get_value() >= 0
        end1 = time.clock()
        time1[i] = end1 - start1
        inds_test = np.where(teQ1 == 1)[0]
        nnz[i] = inds_test.shape[0]

        # indices for true positives
        inds_true = np.where( teY1 == 1 )[0]
        # intersection of true positives and first-stage prediction
        int_result = np.intersect1d(inds_test, inds_true)
        print("first stage nzs:%d,true nzs:%d,intersection:%d" %(inds_test.shape[0],inds_true.shape[0],int_result.shape[0]))
        # recall from first stage:
        r1 = int_result.shape[0] / inds_true.shape[0]
        # precision from first stage:
        p1 = int_result.shape[0] / inds_test.shape[0]
        # accuracy from first stage:
        a1 = np.mean(teY1 == teQ1)
        print("first stage: recall = %f, precision = %f, accuracy = %f" %(r1,p1,a1))

        # only send positive cases from first stage to the second stage:
        teX1 = teX1[inds_test,:]

        # prediction for the second stage
        start1 = time.clock()
        for t in range(n_it):
            teQ2 = predict_second(teX1)
        end1 = time.clock()
        time1[i] += end1 - start1

        # output labels from the cascade
        teY3 = np.zeros(teY1.shape, dtype = int)
        teY3.fill(0)
        teY3[inds_test] = teQ2
        # accuracy of the cascade:
        accuracy1[i] = np.mean(teY1 == teY3)

        inds_second = np.where(teY3 == 1)[0]
        int_result2 = np.intersect1d(inds_second,inds_true)
        print("second stage nzs:%d,true nzs:%d,intersection:%d" %(inds_second.shape[0],inds_true.shape[0],int_result2.shape[0]))
        # recall for the cascade:
        r2 = int_result2.shape[0] / inds_true.shape[0]
        # precision for the cascade:
        p2 = int_result2.shape[0] / inds_second.shape[0]
        print("second stage: recall = %f, precision = %f, accuracy = %f" %(r2,p2,accuracy1[i]))
        # F1 score for the cascade:
        F1[i] = 2*r2*p2/(r2 + p2)

    return time1, accuracy1, F1, nnz
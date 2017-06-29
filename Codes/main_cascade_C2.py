
from __future__ import division
import numpy as np
# from NN_pretraining import NN_pretraining
from NN_pretraining_one import NN_pretraining_one
from cascade_two_stage import cascade_two_stage
from feature_selection import feature_selection
from read_pm2_data import read_pm2_data
from sklearn.cross_validation import StratifiedKFold
from soft_cascade_LR_1LNN import soft_cascade_LR_1LNN
from scipy import stats
import matplotlib.pyplot as plt


# reading the PuffMarker dataset, X: data, Y: labels
X, Y = read_pm2_data()

# number of cross validation folds
n = 8
skf = StratifiedKFold(Y, n_folds=n, shuffle=True)

# time, accuracy, and F1-score for C1:
time1 = np.zeros((n, ))
accuracy1 = np.zeros((n, ))
F1 = np.zeros((n, ))
# time, accuracy, and F1-score for Soft Cascade (SC):
time2 = np.zeros((n, ))
accuracy2 = np.zeros((n, ))
F2 = np.zeros((n, ))
nnz2 = np.zeros((n, ))
# time, accuracy, and F1-score for Firm Cascade (FC):
time3 = np.zeros((n, ))
accuracy3 = np.zeros((n, ))
F3 = np.zeros((n, ))
nnz3 = np.zeros((n, ))

t = 0
for train_idxs, test_idxs in skf:

    trX = X[train_idxs]
    trY = Y[train_idxs]
    teX = X[test_idxs]
    teY = Y[test_idxs]

    # 1LNN
    # parameter alpha of the gating function:
    a = 10
    # number of hidden units in 1LNN:
    K = 10
    # pre-training/initialization of the parameters of 1LNN:
    w_h1, w_o, b1, bo, t1, accuracy1[t], F1[t] = NN_pretraining_one(trX, trY, teX, teY, K)
    time1[t] = t1/10000

    # f_subset = np.arange(19)  # respiration features
    # f_subset = np.arange(19,32,1) # wrist features
    f_subset = [25, 29]  # median of (roll,pitch)
    # f_subset = np.arange(37)  # all the features
    # number of features used in the first stage
    f = 5
    # indicator function whether or not to use z-normalization, set z to 1 when we have 5 features
    z = 1

    # training and testing data in the first stage, trX2 and teX2:
    trX2, teX2 = feature_selection(trX, teX, f_subset, f, z)

    # firm cascade:
    plambda = [0.25]
    t3, accuracy3[t], F3[t], nnz3[t] = cascade_two_stage(trX, trY, teX, teY, trX2, teX2, w_h1, w_o, b1, bo, plambda, a)
    time3[t] = t3/10000

    # soft cascade:
    beta = [0.0001]
    t2, accuracy2[t], F2[t], nnz2[t] = soft_cascade_LR_1LNN(trX, trY, teX, teY, trX2, teX2, beta, K)
    time2[t] = t2/10000

    t += 1

# ploting bar charts with error bars:
w = 0.6
position = [1, 2, 3]
labels = ['1LNN', r'SC$^2$', r'Our$^2$']
colors = list()
colors.append(plt.rcParams['axes.color_cycle'][6])
colors.append(plt.rcParams['axes.color_cycle'][9])
colors.append(plt.rcParams['axes.color_cycle'][2])

accuracy = [np.mean(accuracy1), np.mean(accuracy2), np.mean(accuracy3)]
time = [np.mean(time1), np.mean(time2), np.mean(time3)]
fscore = [np.mean(F1), np.mean(F2), np.mean(F3)]

accuracy_bar = [stats.sem(accuracy1, ddof=0), stats.sem(accuracy2, ddof=0), stats.sem(accuracy3, ddof=0)]
time_bar = [stats.sem(time1, ddof=0), stats.sem(time2, ddof=0), stats.sem(time3, ddof=0)]
fscore_bar = [stats.sem(F1, ddof=0), stats.sem(F2, ddof=0), stats.sem(F3, ddof=0)]

plt.figure(num=1, figsize=(4, 3))
plt.bar(position, accuracy, yerr=accuracy_bar, align='center', width=w, color=colors, error_kw=dict(ecolor='maroon', lw=1, capsize=3, capthick=1))
plt.xticks(position, labels, fontsize=17)
plt.ylabel('Accuracy')
plt.ylim((0.95, 1.00))
plt.grid(b='True', linestyle='--')
plt.savefig('errorbar_accuracy_c2.pdf', bbox_inches='tight')

plt.figure(num=2, figsize=(4, 3))
plt.bar(position, time, yerr=time_bar, align='center', width=w, color=colors, error_kw=dict(ecolor='maroon', lw=1, capsize=3, capthick=1))
plt.xticks(position, labels, fontsize=17)
plt.ylabel('Time')
plt.ylim((0, 3.5/10000))
plt.grid(b='True', linestyle='--')
plt.savefig('errorbar_time_c2.pdf', bbox_inches='tight')

plt.figure(num=3, figsize=(4, 3))
plt.bar(position, fscore, yerr=fscore_bar, align='center', width=w, color=colors, error_kw=dict(ecolor='maroon', lw=1, capsize=3, capthick=1))
plt.xticks(position, labels, fontsize=17)
plt.ylabel('F1')
plt.ylim((0.75, 0.85))
plt.grid(b='True', linestyle='--')
plt.savefig('errorbar_fscore_c2.pdf', bbox_inches='tight')

# checking statistical significance:
(tvalue_accuracy, pvalue_accuracy) = stats.ttest_rel(accuracy2, accuracy3)
(tvalue_fscore, pvalue_fscore) = stats.ttest_rel(F2, F3)
(tvalue_time, pvalue_time) = stats.ttest_rel(time2, time3)


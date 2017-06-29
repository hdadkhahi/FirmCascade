
Firm Cascade.
——————————————————————————

This code implements the firm cascade framework from:

[1] H. Dadkhahi, B. Marlin, “Learning Tree-Structured Detection Cascades for Heterogeneous Networks of Embedded Devices”, to appear in ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2017.
Available online at: https://arxiv.org/pdf/1608.00159.pdf

Setup.
————————————————————————— 

This code requires Theano and Lasagne. For installation instructions check out:
http://deeplearning.net/software/theano/
and
https://lasagne.readthedocs.io/en/latest/

Usage.
————————————————————————— 

This package includes implementation for cascade models C2 to C6 (see Figure 2 in [1]) for the PuffMarker dataset. For each model, the performance of firm cascade has been compared against soft cascade as well as the single-stage model C1.


The main scripts for different models have been listed below:

C2: main_cascade_C2.py
functions: cascade_two_stage.py, soft_cascade_LR_1LNN.py

C3: main_cascade_C3.py
functions: cascade_three_stage.py, soft_cascade_LR_1LNN_2LNN.py

C4: main_cascade_C4.py
functions: cascade_rw.py, soft_cascade_rw.py

C5: main_cascade_C5.py
functions: tree_cascade_noisyAND.py, tree_soft_cascade_noisyAND.py

C6: main_cascade_C6.py 
function: tree_cascade_v1.py, tree_soft_cascade_v1.py


The following functions have been used in different models:

Reading the data: read_pm2_data.py
Functions for classifier models (LR, 1LNN, 2LNN): cascade_functions.py
Function for selecting features of one of the sensors: feature_selection.py
Pre-training functions: NN_pretraining_one.py (1LNN), NN_pretraining.py (2LNN), second_stage_pretraining.py

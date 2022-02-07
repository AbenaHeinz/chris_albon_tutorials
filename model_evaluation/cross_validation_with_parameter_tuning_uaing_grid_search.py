'''
In Machine Learning two tasks are commonly done of teh smae time in data piplines, cross validations and 
(hyper)parametres tuning, Cross validation is the process of training learners using one set of data and 
testing it using a different set. Parameter turning is the process to selecting the values of a model parameters 
that maximize the accuracy of a model  

In this tutorial, we work though and example wich combineds cross validation and parameter tuning using
sci-kit learn. 

Note: This tutorial is based on examples given in the sci-kit learn documentations.
I have combineda few examples in the documentation, simplfied the code and added extensive explainations/code comments  
'''

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import datasets, svm
import matplotlib.pyplot as plt

'''
Create two datasets 

In the code below. we load the digits() dataset. which contains 64 feature variables. Each feature denotes the  
darkness of a pickel in a 8 x 8 image of a handwritten digit. we can see these features for the first observation
'''

# load the digit data
digits = datasets.load_digits()

# vew the features of the first observation 
digits.data[0:1]

'''
The target data is a vector containing the images; true digit, For Example, the First observation 
is a handritten digit for "0"
'''

# Veiw the target of the first observation
digits.target[0:1]

'''
To demostrate cross validation and parametre tuning, first we are going to divide the digit data 
into two datasets called ```data 1``` and ```data 2```. ```data1``` contrains the first 1000 rows,
while ```data2``` contains the remaining ~800 rows. Note that this split is separated to the cross validation
we will conduct and is done purely to demonstrated something at the end of the tutorial. In other words, 
dont worry about ```data 2``` we will come back to it 
'''

# create dataset 1 

data1_features = digits.data[:1000]
data1_target = digits.target[:1000]


# create dataset 2
data2_features = digits.data[1000:]
data2_target = digits.target[1000:]

'''
Create Parameter Candiates 

Before looking for which combonation of parametres values produces the most accurate model,
we must specify the different candidates values we want to try in the code below
we have a number of candidate parameter values including foue different values for C(1, 10, 100, 1000),
two values for gamma(0.001, 0.0001), and two kernels(linear, rbf). The grid search will try all combinations and parameter
values and select the set of parametres which provieds the most accurate model 
'''

parameter_candidates = [
    {"C":[1,10,100,1000], "kernal":["linear"]},
    {"C":[1,10,100,1000], "gamma": [0.001, 0.0001], "kernal": ["rbf"]}
] 

'''
Conduct Grid Search To Find the Parameters Producting Highest Scores

Now we are readt to conduct the grit search using scikit-learn's GridSearchCV which stands for gridsearch cross validation.
By default, the GridSearchCV's cross validation uses 3-fold KFold of StratifiedKFold depending on the situations
'''

# create a classfier object with the classifire and parametre candidates
clf = GridSearchCV(estimatior=svm.SVC(), param_grid = parameter_candidates, n_jobs=1)
    
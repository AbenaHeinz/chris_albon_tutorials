'''
Adaboost 

1. Assign every obsercation xi, an intial weight value, wi=1/n, where n is the total number of observations 
2. Train a "weak" model (most often a decision tree)
3.  For each observation:
    3.1 if predicted incorrectly the wi is increased
    3.2 if pedicted correctly, wi is decreased
4. Train a new weak model where observations with greater weights are given more priority
5. Repeate steps 3 and 4 until observation perfectly predicted or a preset number of trees are trained 
'''

from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

#load iris  
iris = datasets.load_iris()
X = iris.data
y = iris.target

'''
Create Adaboost ca
'''
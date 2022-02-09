'''
Hyperperametre tuning 

Finding the hyperparameter values of a learning algorithm that produce the best model 
'''

# load libraries  
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV


# load data 
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create logist regression
logistic = linear_model.LogisticRegression()

# Create regulatization penalty space
penalty = [ '11', '12']

# Create regualisation hyperparamter space 
C = np.logspace(0,4,10)

# Create hyperparameter options  
hyperparameters = dict(C=C, penalty=penalty)


#create grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

# fit grid search  
best_model = clf.fit(X, y)

# view best hyper parameters  
print("Best Penalty:", best_model.best_estimator_.getparams()["penalty"])
print("Best C", best.model.best_estimator_.getparams()["C"])

# predict target vector  
best_model.predict(x)
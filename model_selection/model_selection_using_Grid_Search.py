'''
Model Selection  

Finding the Machine Learning algorithm and its hyperparametres values that produce the best models

'''

# import libaries  

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# set random search 
 
np.random.seed(0)

# load iris dataset  

iris = datasets.load_iris()
X = iris.data
y = iris.target

'''
Create pipelines whith model selection search space 

Notice that we incude both multiples possinlie learning algorithms  and multiple possibles hyperparameter values  to search over
'''

# create a pipleline  

pipe = Pipeline([("classifier", RandomForestClassifier())])

# creae space of candidate learning algorithms and their hyperparameters 

search_space = [{"classifier":[LogisticRegression()],
                "classifier__penalty":[11, 12],
                "classifier__c": np.logspace(0, 4, 10)},
                {"classifer":[RandomForestClassifier()],
                "classifer__n_estimators":[10,100.1000],
                "classfier__max_features":[1,2,3]}]

# create gridsearch  
clf = GridSearchCV(pipe, search_space, cv=5, verbose=0)

# fit grid search  
best_model = clf.fit(X,y)

#view best model 
best_model.best_estimator_.get_params()["classifier"]

#predoct using best model  
best_model.predict(X)
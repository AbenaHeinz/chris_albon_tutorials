'''
Finding Best Preprocessing Steps During Model Selection  

We have to be careful to properly handle prepropressing when conducting model selection, 
First GridsearchCV uses cross-validation to determin which model has the highest preformance.
However, in cross validation we are in effect pretendiing that the fold heled out as the test set 
is not seen, and thus not the part of fitting any preprocessing steps(e.g. scaling or standardzation)

Secondly some preprocessing methods have there own parameter which is often have to be supplied by the user. 
By including candidates components values in search spaces, they are treated like any ofther hyperparameter
be to searched over 
'''


# load libraries  
from msilib.schema import Feature
import numpy as np
from sklearn import datasets
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# set random seed 
np.random.seed(0)

#load data  
iris = datasets.load_iris()
X = iris.data
y = iris.target


'''
We are include two different preprocessing steps Principal Component Analysis anda k-best feature selection
'''

#create a combined preprocessing object
preprocess = FeatureUnion([("pca", PCA()), ("kbest", SelectKBest("k=1"))])

# create a pipeline 
pipe = Pipeline([("preprocess", preprocess), ("Classifier", LogisticRegression())])

# create space for candidate values  
search_space = [{"preprocess__pca__n_components": [1,2,3],
                "classifier__penalty": [ "11", "12"],
                "classifier__C": np.logspace(0,4,10)}]

# create grid search  
clf = GridSearchCV(pipe, search_space, cv =5, verbose=0, n_jobs=-1)

# fit grid search  
best_model = clf.fit(X,y)

# veiw bets hyper parameters 
print("Best Number of Principle Components:", best_model.best_estimator_.get_params()["preprocess_pca_n_components"])
print("Best Penalty:", best_model.best_estimator_.get_params()["classifier_C"])
print("Best C:", best_model.best_estimator_.get_params()["classifer_C"])
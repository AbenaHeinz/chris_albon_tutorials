# load libaries  
from unicodedata import digit
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# load the digits data sets  
digits = datasets.load_digits()

# create the feature matrixs
X = digits.data

# create teh target vector
y = digits.target

# create standarizer
Standarizer = StandardScaler()

# create logistic regression
logit = LogisticRegression()

# create a pipeline that standardizes , then runs logistic regression 
pipeline = make_pipeline(Standarizer, logit)

# Create K-fold cross validation 
kf = KFold(n_splits=10, shuffle= True, random_state=1)

# Do k-folds cross validation
cv_results = cross_val_score(pipeline, X, y, cv=kf, scoring="accuracy", n_jobs = 1)

# caluclate mean 
cv_results.mean()
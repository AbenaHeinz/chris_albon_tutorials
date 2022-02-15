'''
Recall 

Recall is all about the real positive  

True Postive/ (True Positive + False Negative)

Recall is the ability of the  classifier to find positive examples.
If we wanted to be centain to find all postive examplesm we could maximsze recall 
'''

# load libaraies  
from sklearn.model_selection import  cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


# generate features matrixs and target_vectors  
X,y = make_classification( n_samples = 10000, n_features = 3, n_informative = 3, n_redundant= 0, n_classes = 2, random_state=1)

# create logistic regression  
logit = LogisticRegression()

#create cross-validation model using  precision  
cross_val_score(logit, X, y,scoring = "recall")

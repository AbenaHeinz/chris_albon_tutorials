'''
Precision

Precision is the ability a classifier to not label a true negative observations as positive  

true positive /(true postive + false postive)
'''

# Load libraries
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

#Generate feature matrix and target vectors  
X, y = make_classification(n_samples = 10000, n_features = 3, n_informative = 3, n_redundant = 0, n_classes = 2, random_state = 1)

# create logicistic regression  
logit = LogisticRegression()

# Cross-validate model using precision 
cross_val_score(logit, X, y, scoring = "precision")


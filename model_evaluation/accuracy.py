'''
Accuracy  

A common metric in classification. Fauls when we have highly imbalanced classes. 
In those cases FM is more appropriated
'''

# load libaries  

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


# generate features and target vector

X, y = make_classification(n_samples = 100000, n_features = 3, n_informative = 3, n_redundant = 0, n_classes = 2, random_state = 1)

# create logistic regression
logit = LogisticRegression()

# cross-valiadte modles uring accuracy  
cross_val_score(logit, X, y, scoring="accuracy")

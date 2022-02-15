'''
Validation Curve 

Validation Curves visualise the preformance metric ofver a range of values for some hyperparameter.
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve

# load digits data 
digits = load_digits()
# create feature matrix and target vector
X, y = digits.data, digits.target

#create a range of values for the parameters  
param_range = np.arange(1, 250, 2)

# calculate accurary on training and trst sets using range of parameter values
train_scores, test_scores = validation_curve(RandomForestClassifier(), X, y, param_name="n_estimators", param_range= param_range, cv= 3, scoring= "accuracy", n_jobs = -1)

# calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# calculate mean and standard deviation from test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot accuracy bands for trianing and test set  
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color= "grey")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

#create plot
plt.title("Validation Curves With Random Forest")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc= "best")
plt.show()
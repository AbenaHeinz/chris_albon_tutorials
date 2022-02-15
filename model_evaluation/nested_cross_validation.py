'''
Often we want to tune the parametres of a model(for example, C in a support vector machine).
That is, we want to find the balues of a parameter that minimises otr loss function, The best way  
To do is is cross validaation. 

1. Set the parameter you want to to tune to some value. 
2. Split your data into K "folds"(sections)
3. Train your model  using K-1 folds using the parameter values  
4. Test your model on the remaining foldes
5. Repeat steps 3 and 4 so that every fold is the test data once  
6. Repeat steps 1 to 6 for every possible parameter
7. Repeat the parametre that prodice the best result

However, as Cawley and Talbot point out in their 2010 paper, since we use the test set to bot select the values 
of the parameter and evaluate the model, we risk optimistical blasing our model evaluation. For this reason,  
If a test set is used to select model parameters  then we need a different test set to get and unbiased
evaluation of a selected model  

Once way to overcome this probelm is to have nested cross validiation. First, and inner cross validation is used
to tune the parameters and select the best model. Second, an outer cross validations is used to evaluate the model
selected by the inner cross validation.
'''


# load required packages  
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC

# load the data  
datasets = datasets.load_breast_cancer()

# crete X from the features
X = datasets.data

# create y from the targets  
y = datasets.target

# Create a scalar object  
sc = StandardScaler()
X_std = sc.fit_transform(X)

'''Create Inner Cross validation (For Parameter Tuning)
This is our inner cross validation. We will use this to hunt the best paramentre for C,  the penalty for misclassifying 
a data point. GridsearchCV - will conduct steps 1-6 list at the top of this tutorial
'''

# create a list of 10 candidates values for the C parameter 
C_candidates = dict(C=np.logspace(-4, 4, 10))

# create a gridsearch object with the supplort vector classifier and the C value candidates 
clf = GridSearchCV(estimator=SVC(), param_grid=C_candidates)

'''
The code below isn't necessary for parameter turning using nester cross validation.=, however to demostrate that our
inner cross validation grid search can find the best value for the parameter C, we ill run it once here 
'''

# Fit the cross validated grid search on the data 
clf.fit(X_std, y)

# show the best value for C
clf.best_estimator_.C 

# Create Outer Cross Validation (For Model Evaluation)
'''
With our inner cross validation constructed, we can use cross_val_score to evaluate the model with a second (outer)
cross validation. 
The code below splits the data into three folds,  running in the inner cross validation on two of the folds (merged together)
and then evaluating the model on the third fold . This is reeared three times so that every fold is used for testing once
'''
cross_val_score(clf, X_std, y)

'''
Each the values above is an unbiased evaluation of the model's accuracy. once for eacg of the three test folds
Averaged together, they would represent the avarage accruacy of the model founds in the inner cross validated grid search
'''


# load libraries  
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV


# load iris  
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create logistice regression 
logistic = linear_model.LogisticRegression(C=1, penalty='l1', solver='liblinear')

# create regulilization penalty space  
penalty = ['11', '12']

# Create regualization hyperparameter distribution using uniform distribution  
C = uniform(loc=0, scale=4)

# create hyer parameter options
hyperparameters = dict(C=C, penalty = penalty)

# Create Random Search 5-Fold cross validation and 100 iterations 
clf = RandomizedSearchCV(logistic, hyperparameters, random_state=1, n_iter=100, verbose=0, n_jobs=-1)

# Fit Randomized search
best_model = clf.fit(X,y)

#veiw best hyperparameter
print("Best Penalty:", best_model.best_estimator_.get_params()["penalty"])
print("Best C:", best_model.best_estimator_.get_params()["C"]) 

# predict target vectors 
best_model.predcit(X)
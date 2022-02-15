# import require packages  
import numpy as np
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler


# Load the breast cancer dataset  
dataset = datasets.load_breast_cancer()

# create X from the dataset features  
X = dataset.data

# create y from the dataset's output 
y = dataset.target

# create an scaler object  
sc = StandardScaler()

# create PCA object  
pca = decomposition.PCA()

# Create a logistic regression with an L2 penalty 
logistic = linear_model.LogisticRegression()

# create a pipelines of three steps, First standardize the data  
# Second, transform the data with PCA
# third, train a logistic regression on the data  

pipe = Pipeline(steps = [("sc",sc), ("pca", pca), ("logistic", logistic)])


# create a list of a sequence of integers from 1 to 30 (the number of features in X + 1)
n_components = list(range(1, X.shape[1]+1,1))

# create a list of values of the regualization parametre  
C = np.logspace(-4, 4, 50)

# Create a list of options for the regualization penalty 
penalty = ['11', '12']

# create a dictionary of all of the parametre options 
# note has your can access teh parametres of steps of a pipeline using '__'
parametres = dict(pca__n_components = n_components, logistic__C = C, logistic__penalty = penalty)

# Create a grid search object  
clf = GridSearchCV(pipe, parametres)

# fit the grid seach  
clf.fit(X, y)

# view the best parameters 
print("Best Penalty:", clf_best_estimator_.get_params()["logistic__penalty"] )
print("Best C:", clf_best_estimator_.get_params()["logistic__C"])
print("Best Number of Components:", clf_best_estimators_.getparams()["pca__n_components"])

# Fit the grid search ysing 3-fold cross validation 
cross_val_score(clf, X,y)
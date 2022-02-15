from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model
import warnings

# suppress annouing part harmless warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# generate features matrix, target vectors, and the true coefficents

X,y = make_regression(n_samples=10000, n_features=100, n_informative=2, random_state=1)

# create a linear regression  
ols = linear_model.LinearRegression()

# create recursive features eliminated that scores features by means square errors 
rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")

#Fit recursive feature eliminator  
rfecv.fit(X, y)

#recursive feature eliminator 
rfecv.transform(X)

# Number of the best feature
rfecv.n_features_
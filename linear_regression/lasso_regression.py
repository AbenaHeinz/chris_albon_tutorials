#load libaries  
from pyexpat.errors import XML_ERROR_ASYNC_ENTITY
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# load data  
boston = load_boston()
X = boston.data
y = boston.target

# standardize feature  
scaler = StandardScaler
X_std = scaler.fit_transform(X)

'''
The hyperparameter alpha, lets us control how much we peanilise teh coeffeicentm with higer values of alpha
creating simper modelers. The ideal value of alpha should be tuned like any other hyperparameter. In scikit-learn
alpha is set using the alpha parameter
'''

# create lasso regression with alpha value
regr = Lasso(alpha=0.5)

# fit the lasso regression model  
model = regr.fit(X_std, y)
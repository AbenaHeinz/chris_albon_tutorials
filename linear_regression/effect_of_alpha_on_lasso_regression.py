'''
Often we want to conduct a process called regualization, wherein we penalize the number of feature
in a model in order to keep the most important feature. This can be particularly important when you have a 
dataset with 100,000+ features

Lasso Regreesion is a common modeling technique to do regualization, the math behind it is pretty intresting 
but practically, what yo need to know is that Lasso regressuon come with a parameter alpha and the higher
the alpha, the most feature coefficiants are zero 

That is when alpha is O, Losso Regressiion produces the same coefficcian as linea regression. When alpha
is very very large all of the coefficients are zero

In this tutorial, I ran three lasso regressionsm with varying levels of alpha, and show the resulting effect on the coefficients

'''

# load libraries  
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
import pandas as pd
from sqlalchemy import column


# load data  
boston = load_boston()
scaler = StandardScaler()
X = scaler.fit_transform(boston["data"])
Y = boston["target"]
names = boston["feature_names"]

# create a function called lasso 

def lasso(alphas):
    df = pd.DataFrame()
    df["Feature Names"] = names
    for alpha in alphas:
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, Y)
        column_names = "Alpha = %f" % alpha
        df[column_names] = lasso.coef_
    return df

# run the function called Lasso
lasso([.0001, .5, 10])

# Notice that as the alpha value increase more features have a coefficicent of 0

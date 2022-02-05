import pandas as pd
from sklearn.datasets import make_regression

# Create Simulated Data  
'''Generate features, outputs and true coefficents for 100 samples'''
features, outputs, coef = make_regression(n_samples= 100, n_features= 3, n_informative=2, noise=0.0, coef = True)

# view simulated data 
'''View the features for the first five rows'''
pd.DataFrame(features, columns=["Store_1", "Store_2", "Store_3"]).head()

'''view the ouputs of the first five rows'''
pd.DataFrame(output, columns=["Sales"]).head()

'''View the actual, true coefficents using to generate the data '''

pd.DataFrame(coef, columns=["True Coefficient Values"])


'''One-hot encoding allows us to turn nomnial catergorical data into features with numerical values,
while not mathematically imply, any ordinal relationship between the classes'''

# load libraries  
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Create Numpy array  
x = np.array([["Texas"], ["California"], ["Texas"], ["Delaware"], ["Texas"]])

# one-hot encode data method 1 

# create LabelBinzarizer object  
one_hot = OneHotEncoder

# one-hot encoder data 
one_hot.fit_transform(x)

# view classes  
one_hot.categories


# One-hot encode data method 2

#dummy features
pd.get_dummies(x[:0])
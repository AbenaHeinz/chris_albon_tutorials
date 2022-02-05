'''
Standardization  

Standardization is a common scaling method; represents the number of atandard deviations.
Each value is from the mean value. It rescales a feature to have a mean of 0 and unit variance.
'''

# load libraries 

from sklearn import preprocessing
import numpy as np

# create a feature 
x = np.array([[-500.5], [100.1], [0], [100.1], [100.9]])

# create  scalar
scaler = preprocessing.StandardScaler()

# Transform the feature 
standardized = scaler.fit_transform(x)

# show feature
standardized
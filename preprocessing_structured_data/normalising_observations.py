'''Normalising Observations  

Rescaling the feature values of each obsercation so that they have a unit norm
Two common norm values are L1 and L2
'''

# Load Libraries  

from sklearn.preprocessing import Normalizer
import numpy as np

# Create Feature Matrix

X = np.array([[0.5, 0.5], [1.1, 3.4], [1.5, 20.2], [1.63, 34.3], [10.9,3.3]])

'''Normalize Observations 
Normalizer rescales the values on individual observations to have unit norm (the sum of their lenghts is one)'''

# create the normaliser

normalizer = Normalizer(norm='12')

# Transform Feature Matrix
normalizer.transform(X)
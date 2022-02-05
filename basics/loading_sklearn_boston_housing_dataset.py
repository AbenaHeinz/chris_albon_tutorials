#load libraries 
from sklearn import datasets
import matplotlib.pyplot as plt

# load digits datasets 

boston = datasets.load_boston()

# Create feature matrix
X = boston.data

# create target vector
y = boston.target 

# View the first observations feature values
X[0]

'''As you can see the feature are not standardized. This is more easily seen if we display the values as decimals'''

# display each feature value of the first observation as floats  
["{:f}".format(x) for x in X[0]]

'''Therefore it is often benefitical and/or required to standarize the values of the features'''
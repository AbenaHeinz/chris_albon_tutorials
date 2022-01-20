'''Downsampling 
a stratagey to handle imbalanced classes by creating a random subest of the majority of equal size to the miniority class
in downsampling we randomly samples without replacement from the majority class (i.e. the lasss with more obsercations)
to creae a new subset of obsercations equal in size to the minority class'''

# import libaries 
import numpy as np
from sklearn.datasets import load_iris

# load iris dataset  
iris = load_iris
# create a feature matrix
X = iris.data
#create a target vector
y = iris.target

#make iris dataset imbalanced 

# remove first 40 observations  
X = X[40:,:]
y = y[40:]

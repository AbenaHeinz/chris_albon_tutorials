'''Downsampling 
a stratagey to handle imbalanced classes by creating a random subest of the majority of equal size to the miniority class
in downsampling we randomly samples without replacement from the majority class (i.e. the lasss with more obsercations)
to creae a new subset of observations equal in size to the minority class'''

# import libaries 
import numpy as np
from sklearn.datasets import load_iris

# load iris dataset  
iris = load_iris()
# create a feature matrix
X = iris.data
#create a target vector
y = iris.target

#make iris dataset imbalanced 

# remove first 40 observations  
X = X[40:,:]
y = y[40:]

# create binary targets vectors indicating if class 0 
y = np.where((y == 0), 0, 1)

# Look at the imbalanced target vector 
y

# Downsample Majority Class to Match Minority Class

#indices of each class observations
i_class0 = np.where(y == 0)[0]
i_class1 = np.where(y == 1)[0]

#Number of obsevations in each class 
n_class0 = len(i_class0)
n_class1 = len(i_class1)

# for every observation of class 0, randomly sample from class 1 without replacement 
i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace = False)

# join together class 0's target vextor 
np.hstack((y [i_class0], y[i_class1_downsampled]))
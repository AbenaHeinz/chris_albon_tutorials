from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the iris dataset 

iris = datasets.load_iris()

# create a variable for the feature data  
X = iris.data

# create a variable for the target data  

y = iris.target

# Randomly split the data onto for new datasets, training feature, training outcomes test features
# and test outcome. Set the size of the test data to be 30%of the full dataset  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Load the standard scaler  
sc = StandardScaler()

# Compute the mean and the standard deviation based on the training data  
sc.fit(X_train)

#scale the trianing data to be of mean 0 and of unit variance  
X_train_std = sc.transform(X_train)

# Scale the test data to be of mean 0 and of unit variance  
X_test_std = sc.transform(X_test)

# Feature test data, non-standardized 
X_test[0:5]



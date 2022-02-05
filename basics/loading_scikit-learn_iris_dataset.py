# load libraries 

from sklearn import datasets
import matplotlib as plt

# Load Iris Data  

''' The Iris Flower dataset is one of the most famous databases for classification. 
It contains three classes (i.e. three species of flowers) with 50 observations per class'''

# load digits dataset

iris = datasets.load_iris()

# create a feature matrix

X = iris.data

# Create target vector
y = iris.target

# view the first observation's feature values 
X[0]

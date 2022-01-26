'''
Upsampling 

A strategy to handle imbalanced classes by repeatedly sample with replacement 
from the miniority class to make it of equal size as the majority class

In upsampling, for ever obsercation in the majority class, we randomly select and observation from the minority classe 
with replacements.  The end result is the same number of observations from the minority and majority class
'''

#Import libaries  
from json import load
from tkinter import Y
import numpy as np
from sklearn.datasets import load_iris

# load iris dataset 
iris = load_iris()
X = iris.data
y = iris.target

# make iris dataset imbalanced 

# remove first 40 observations
X = X[40:,:]
y = y[40:]

# create binary target verxtors indicating if class 0
y = np.where((y==0), 0, 1)

# look at the imbalanced targets  
y 

# Upsampiling Minority Class to Match Majority

# Indicies of each class' observations 
i_class0 = np.where(y==0)[0]
i_class1 = np.where(y==1)[0]

# Number of observations in each class  
n_class0 = len(i_class0)
n_class1 = len(i_class1)

# for every observation in class 1, randomly sample from class 0 with replacement  
i_class0_unsampled = np.random.choice(i_class0, size=n_class1, replace= True)

# join together clas 0's upsampled target vector with class 1's targets vectors  
np.concatenate((y[i_class0_unsampled], y[i_class1]))
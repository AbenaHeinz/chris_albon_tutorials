from sklearn.preprocessing import Binarizer, binarize
import numpy as np

#create features 
age = np.array([[0], [12], [20], [36], [65]])

#option 1: Binarize Feature

#create binarizer
binarizer = Binarizer(18)

#transform featre
binarizer.fit_transform(age)

#option 2@ break up features into bins  
#bin feature

np.digitize(age,bins=[20,30,40])
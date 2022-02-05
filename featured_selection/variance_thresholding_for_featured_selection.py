'''
variance thresholding  

- motivated by the idea that low variance features contains  less information  
- Calculate variance of each feature, then drops features with variance below some threshold 
- make sure that features have the same scale 
'''


from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold

#load iris data  
iris = datasets.load_iris()

# create features ad targets  
x = iris.data
y = iris.target

# create variancethreshold object with a variance with a threshold of 0.5
thresholder = VarianceThreshold(threshold=.5)

# conduct variance thresholding
x_high_variance = thresholder.fit_transform(x)

#view first five rows with features with variances above threshold
x_high_variance[0:5]
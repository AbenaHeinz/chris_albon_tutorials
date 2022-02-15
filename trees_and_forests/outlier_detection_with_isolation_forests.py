'''
An isolation forest is comprised of many of a special kind of trees. in each not the tress,
a random feature is split into a random point (for example,  if values in a feature range between 0 and 100, a random split might be 34)
This process contunues until every data point is isolated in its own brand. This is repreated until the end results is a forest of trees

The intution that is isolating a non-outlier data point will require many splits to be isolated because it is very similar to other data points.
On the flip side, an outlier data point will require few slits to be isolated because it is very disimlar to the other data points 
'''

from unittest import IsolatedAsyncioTestCase
from sklearn import cluster
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
import numpy as np

# make_blobs with a single centre will create a single cluster of data  

# make the feature (X) with 300 samples, two feature variables, three clusters aith .5 cluster standard deviation, shuffled

X, _ = make_blobs(n_samples=200, n_features=2, centers=1, cluster_std=0.5, shuffle = True)

# veiw the two features  
X

# create an outlier datapoint  
outlier = [[100,100]]

#concatinates the  outlier with X
X_with_outlier = np.concatenate((outlier,X))

# set the isolation forest, randomly sample observation for each tree with replacement, number of trees,the contamination and behaviour variables are unnecessary but added to avode a depreciation errors
clf = IsolationForest(bootstrap=True, n_estimators=100, contamination="auto", behaviour="new")

# train the isolation forest  
clf.fit(X_with_outlier)
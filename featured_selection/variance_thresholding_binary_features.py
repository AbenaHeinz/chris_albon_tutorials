from sklearn.feature_selection import VarianceThreshold

#create feature matrix
# Feature 0: 80% class 0
# Feature 1: 80% class 1
# Feature 2: 60% class 0, 40% class 1

X =[[0,1,0], [0,1,1],[0,1,0], [0,1,1], [1,0,0]]

''' Conduct the variance thresholding  

In binary features, (i.e Bernoulli random variables), variance is calulated as:
where p is the proportions of obervations of class 1 and Therefore by setting p
we can remove features where the vast majority of observatios are one class
'''

# Run threshold by variance

threshold = VarianceThreshold(threshold=(.75 * (1 - .75)))
threshold.fit_transform(X)

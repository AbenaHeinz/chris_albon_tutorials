from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Make data 

# Make the features (X) and the output (Y) with 200 samples 

X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.5, shuffle=True)

# create a scatterplot for the first and second features
plt.scatter(X[:,0], X[:,1])

#show the scatterplot
plt.show()
# load libaries 
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Make data 
'''Make the features (X) and the output (y) with 200 samples'''
X,y = make_blobs(n_samples= 200, 
                #two feature variables 
                n_features=2,
                # three clusters 
                centers=3, 
                # with .5 cluster standard deviations
                cluster_std=0.5,
                #shuffled
                shuffle=True)
# View Data  
'''Create a scatterplot of the first and second features'''
plt.scatter(X[:,0], X[:,1])

# show the scatter plot 
plt.show()
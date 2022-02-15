from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd

# Make simulated feature matrixs
X, _ = make_blobs(n_samples = 50, n_features = 2, centers = 3, random_state = 1)
# create a dataset  
df = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])

# make k-means clusterer
clusterer = KMeans(3, random_state=1)
#fit the cluster  
clusterer.fit(X)

#Predict values  
df["group"] = clusterer.predict(X)

#first five observation  
df.head(5)

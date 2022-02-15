'''
Principle Component Analysis(PCA)

PCA projects the features onto the principal components. The Motivation is to reduce the features dimensionally while
only loosing a small amount of information.
'''

# load libaries  
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

# load the data  
digits = datasets.load_digits()

# Standardize the feature matrix
X = StandardScaler().fit_transform(digits.data)

# create a PCA that will retain 99% of the variance 
pca = PCA(n_components=0.99, whiten=True)

# Conduct PCA
X_pca = pca.fit_transform(X)

#show results  
print("Original number of features:", X.shape[1])
print("Reduce number of features:", X_pca.shape[1])


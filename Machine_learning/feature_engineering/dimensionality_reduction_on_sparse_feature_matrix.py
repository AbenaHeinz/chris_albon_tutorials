import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np

# load the data  
digits = datasets.load_digits()

# Standarize the feature matrix 
X = StandardScaler().fit_transform(digits.data)

# Make sparse matrix
X_sparse = csr_matrix(X)

# create a TSVD(Truncted Singular Value Decomposition)
tsvd = TruncatedSVD(n_components = 10) 

# conduct TSVD on sparse matrix  
X_sparse_tsvd = tsvd.fit(X_sparse).transform(X_sparse)

# show results  
print("Original number of feature:", X_sparse.shape[1])
print("Reduced number of features:", X_sparse_tsvd.shape[1])

#Sum of first three components' explained variance ratios 
tsvd.explained_variance_ratio_[0:3].sum()

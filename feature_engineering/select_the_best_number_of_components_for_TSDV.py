# load libraries 

from base64 import standard_b64encode
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np

# load the data  
digits = datasets.load_digits()
# Standardize the  feature matrixs
X = StandardScaler().fit_transform(digits.data)
# make sparse matrix
X_sparse = csr_matrix(X)

# create and run a TSVD with one less tahn the number of features  
tsvd = TruncatedSVD(n_components=X_sparse.shape[1]-1)
X_tsvd = tsvd.fit(X)

# list explained variances
tsvd_var_ratios =  tsvd.explained_variance_ratio_


'''
Create a function calculating number of components required to pass threshold
1. Create function
2. Set initial variance explained so far (total variance)
3. Set Inital number of features(n_components)
4. for the explained variance of each feature,  add the explained variance to the total, add one to the number of components  
5. if we reach our goal level for explained variance, end the loop
6. return the number of components  
'''
def select_n_components(var_ratio, goal_var: float) -> int:
    total_variance = 0.0
    n_components = 0 
    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_var:
            break
    return n_components

# run function
select_n_components(tsvd_var_ratios, 0.95)


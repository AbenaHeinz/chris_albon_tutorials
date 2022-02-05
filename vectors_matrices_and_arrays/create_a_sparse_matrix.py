#load libraries

import numpy as np
from scipy import sparse

# create a dense matrix 
matrix = np.array([[0,0], [0,1], [3,0]])

# convert to Sparse Matrix 
# create compressed spase row (CRS) matrix

matrix_sparse = sparse.csr_matrix(matrix)
matrix_sparse